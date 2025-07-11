from gurobipy import Model, GRB, quicksum
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import numpy as np
from agps_config import (
    DEFAULT_SFC_AGPS,
    DEFAULT_SPEED_AGPS,
    DISTANCE_MATRIX,
    DEFAULT_STARTUP_TIME,
    DEFAULT_WARMUP_TIME,
    DEFAULT_BUFFER_AGPS,
)
from agps_funs import fuelMESengine_df, fuelMESapu_df
from datetime import timedelta
import re
from tqdm import tqdm


# Driving distances from Runway to Stand Areas in kilometers [km]
# All runways to unknwon: 2.8km (which is equal to the average of the distances to the other stand areas)
# distance_matrix_LSZH = pd.DataFrame(
#     data=[  # A_N    AB_C    B_S     C,      D,      E,      F,      G,      H,      I,      T,      GAC     unknown
#         [4.4, 3.8, 3.7, 3.5, 3.3, 5.3, 4.6, 2.8, 4.4, 4.7, 3.3, 5.0, 2.8],  # RWY 10
#         [3.6, 3.6, 3.8, 4.4, 4.2, 2.1, 3.7, 5.1, 3.6, 3.9, 4.6, 4.1, 2.8],  # RWY 16
#         [1.3, 1.5, 1.5, 2.0, 1.8, 2.0, 0.6, 2.7, 1.2, 1.5, 2.2, 0.4, 2.8],  # RWY 28
#         [2.4, 2.5, 2.7, 3.3, 3.0, 1.1, 1.8, 3.8, 2.4, 2.7, 3.3, 1.6, 2.8],  # RWY 32
#         [2.2, 1.7, 1.4, 1.3, 1.0, 3.0, 2.4, 0.5, 2.2, 2.4, 1.0, 2.7, 2.8],  # RWY 34
#     ],
#     index=["10", "16", "28", "32", "34"],  # Takeoff runways as rows
#     columns=[
#         "A North",
#         "AB Courtyard",
#         "B South",
#         "C",
#         "D",
#         "E",
#         "F",
#         "G",
#         "H",
#         "I",
#         "T",
#         "GAC",
#         "unknown",
#     ],  # Stand Areas as columns
# )


def get_df_movements(filepath: str) -> pd.DataFrame:
    """
    Function to read the pickle file and return a DataFrame.
    """
    # Read df_movements (df_movements is the output of agps_proc.ipynb)
    df_movements = pd.read_pickle(filepath)

    # Filter for takeoff only
    df_movements = df_movements.query("isTakeoff", engine="python")

    # Get rid of departures classified as runway 14 -> These are outliers (no aircraft take off on runway 14 at ZRH in real life)
    df_movements = df_movements.query('takeoffRunway!="14"')

    # Add date column
    df_movements["date"] = df_movements["lineupTime"].dt.date

    #### Stand areas
    # Overwrite stand_area == "unkown" with the first character of the parking_position (if available)
    df_movements["stand_area"] = df_movements["stand_area"].replace("unkown", "unknown")

    # Define mask: not NA and does not start with 'V'
    mask = df_movements["parking_position"].notna() & ~df_movements[
        "parking_position"
    ].str.startswith("V")

    # Overwrite stand_area with first letter of parking_position where mask is True
    df_movements.loc[mask, "stand_area"] = df_movements.loc[
        mask, "parking_position"
    ].str[0]

    #### Calculate "start Movement" time
    # Calculate startMovement as the earlier of the two times
    df_movements["startMovement"] = df_movements[["startTaxi", "startPushback"]].min(
        axis=1, skipna=True
    )

    #### Add Airline column based on callsign
    def contains_letters_and_numbers(s):
        return bool(re.search(r"[A-Za-z]", s)) and bool(re.search(r"\d", s))

    # Apply the function to the 'callsign' column
    df_movements["contains_letters_and_numbers"] = df_movements["callsign"].apply(
        contains_letters_and_numbers
    )

    # Add the 'airline' column based on the condition
    df_movements["airline"] = df_movements.apply(
        lambda row: row["callsign"][:3]
        if row["contains_letters_and_numbers"]
        else "unknown",
        axis=1,
    )

    # Drop not required columns
    df_movements = df_movements.drop(
        columns=[
            "MESengine180",
            "MESapu180",
            "normTAXIengine180",
            "min_compute_gs_diff",
            "max_compute_gs_diff",
            "max_cumdist_diff",
            "first_altitude",
            "normTAXIengine",
            "extAGPSapu",
            "extAGPStug",
            "contains_letters_and_numbers",
        ]
    )

    # Convert all relevant timestamp columns to timezone-naive
    df_movements["startTaxi"] = pd.to_datetime(
        df_movements["startTaxi"]
    ).dt.tz_localize(None)
    df_movements["lineupTime"] = pd.to_datetime(
        df_movements["lineupTime"]
    ).dt.tz_localize(None)
    df_movements["startPushback"] = pd.to_datetime(
        df_movements["startPushback"]
    ).dt.tz_localize(None)
    df_movements["date"] = pd.to_datetime(df_movements["date"]).dt.tz_localize(None)

    return df_movements


def calculate_normal_taxi_fuel(
    df_movements: pd.DataFrame,
    startupTime=DEFAULT_STARTUP_TIME,       # Startup time of one single engine in seconds
    warmupTime=DEFAULT_WARMUP_TIME,         # Warmup time (for the entire flight) in seconds
    sfc_agps=DEFAULT_SFC_AGPS,              # Specific fuel consumption AGPS [kg/s]
    outputColName="F_i_norm",               # Output column name for normal fuel consumption
) -> pd.DataFrame:
    # Number of engines
    nEngines = df_movements["nEngines"].fillna(2).astype(int)

    # Duration of pushback
    dur_PB_raw = df_movements["startTaxi"] - df_movements["startPushback"]
    dur_PB = np.where(
        df_movements["startPushback"].isna(), 0, dur_PB_raw.dt.total_seconds()
    )
    dur_PB = np.clip(dur_PB, a_min=startupTime * nEngines, a_max=None)

    # Fuel consumption of PB truck during Pushback
    F_PBtug = dur_PB * sfc_agps  # dur_PB is in seconds, sfc_tug should be in kg/s

    # APU normal mode on stand/during pushback
    dur_APUnorm = np.clip(dur_PB - startupTime * nEngines, a_min=0, a_max=None)
    F_APUnorm = (
        dur_APUnorm * df_movements["APUnormalFF"] / 3600
    )  # FF in kg/h, duration in sec

    # APU & Engine during MES & WUP
    df_movements = fuelMESengine_df(
        df_movements=df_movements,
        startupTime=startupTime,
        warmupTime=warmupTime,
    )
    df_movements = fuelMESapu_df(
        df_movements=df_movements,
        startupTime=startupTime,
        warmupTime=warmupTime,
    )
    F_mw = df_movements["MESapu"] + df_movements["MESengine"]

    # Duration of conventional taxi
    dur_convTaxi = (
        df_movements["lineupTime"] - df_movements["startTaxi"]
    ).dt.total_seconds() - warmupTime
    dur_convTaxi = dur_convTaxi.clip(lower=warmupTime)  # keep as Series to use .clip()

    # Fuel consumption of engines during conventional taxi
    F_convTaxi = (
        dur_convTaxi * df_movements["nEngines"] * df_movements["engIdleFF"]
    )  # FF in kg/s

    # Final normal fuel consumption of flight i
    df_movements[outputColName] = F_mw + F_convTaxi + F_PBtug + F_APUnorm

    return df_movements


def calculate_agps_taxi_fuel(
    df_movements: pd.DataFrame,
    startupTime=DEFAULT_STARTUP_TIME,       # Startup time of one single engine in seconds
    warmupTime=DEFAULT_WARMUP_TIME,         # Warmup time (for the entire flight) in seconds
    sfc_agps=DEFAULT_SFC_AGPS,              # Specific fuel consumption AGPS [kg/s]
    outputColName="F_i_agps",               # Output column name for AGPS-assisted fuel consumption
) -> pd.DataFrame:
    # Number of engines
    nEngines = df_movements["nEngines"].fillna(2).astype(int)

    # Duration of AGPS-assisted taxi
    df_movements["duration_AGPS"] = (
        (df_movements.lineupTime - df_movements.startMovement).dt.total_seconds()
        - startupTime * nEngines
        - warmupTime
    ).clip(lower=0)

    # APU & Engine during MES & WUP
    df_movements = fuelMESengine_df(
        df_movements=df_movements,
        startupTime=startupTime,
        warmupTime=warmupTime,
    )
    df_movements = fuelMESapu_df(
        df_movements=df_movements,
        startupTime=startupTime,
        warmupTime=warmupTime,
    )
    F_mw = df_movements["MESapu"] + df_movements["MESengine"]

    # Fuel Consumption of APU during AGPS-assisted Taxi
    F_APUagps = df_movements["duration_AGPS"] * df_movements["APUnormalFF"] / 3600

    # Fuel consumption of tug during AGPS-assisted taxi
    F_agps = (df_movements["duration_AGPS"] + startupTime * nEngines) * sfc_agps

    # Final AGPS-assisted fuel consumption of flight i
    df_movements[outputColName] = F_mw + F_APUagps + F_agps

    return df_movements


def calculate_driving_distance(
    runway: str,
    stand_area: str,
    distance_matrix: pd.DataFrame = DISTANCE_MATRIX,
    dist_default=2.8,
) -> float:
    """
    Retrieve distance between runway and stand area from the distance matrix.
    If the combination does not exist, return a default value.
    """
    try:
        return distance_matrix.loc[runway, stand_area]
    except KeyError:
        return dist_default  # Default distance if not found


def filter_flights(
    subset_df: pd.DataFrame,
    allowed_airlines: list = [],
    allowed_aircraft_types: list = [],
    allowed_stand_rwy_combinations: list = [],
    allowed_time_windows: list = [],
) -> pd.Series:
    """
    Filters flights based on the given criteria.

    Args:
        subset_df (pd.DataFrame): The DataFrame containing flight data.
        allowed_airlines (list): List of allowed airlines.
        allowed_aircraft_types (list): List of allowed aircraft types.
        allowed_stand_rwy_combinations (list): List of allowed stand-runway combinations.
        allowed_time_windows (list): List of allowed time windows as (weekday, start_time, end_time).

    Returns:
        pd.Series: A boolean mask indicating valid flights.
    """
    valid_flights_mask = pd.Series([True] * len(subset_df), index=subset_df.index)
    # valid_flights_mask = pd.Series([True] * len(subset_df))

    # Filter by allowed airlines
    if allowed_airlines:
        valid_flights_mask &= subset_df["airline"].isin(allowed_airlines)

    # Filter by allowed aircraft types
    if allowed_aircraft_types:
        valid_flights_mask &= subset_df["typecode"].isin(allowed_aircraft_types)

    # Filter by allowed stand-runway combinations
    if allowed_stand_rwy_combinations:
        valid_flights_mask &= subset_df.apply(
            lambda row: (row["stand_area"], row["takeoffRunway"])
            in allowed_stand_rwy_combinations,
            axis=1,
        )

    # Filter by allowed time windows
    if allowed_time_windows:
        valid_flights_mask &= subset_df.apply(
            lambda row: any(
                row["startMovement"].weekday() == day
                and pd.to_datetime(start_time).time()
                <= row["startMovement"].time()
                <= pd.to_datetime(end_time).time()
                for day, start_time, end_time in allowed_time_windows
            ),
            axis=1,
        )

    return valid_flights_mask


def optimize_day_MILP_gurobi(
    # date: pd.Timestamp,
    subset_df: pd.DataFrame,
    n_tugs: int = 1,
    distance_matrix: pd.DataFrame = DISTANCE_MATRIX,  # [km]
    buffer_time: pd.Timedelta = DEFAULT_BUFFER_AGPS,
    agps_speed: float = DEFAULT_SPEED_AGPS,  # [km/h]
    agps_sfc: float = DEFAULT_SFC_AGPS,  # [kg/h]
    allowed_airlines: list = [],
    allowed_aircraft_types: list = [],
    allowed_stand_rwy_combinations: list = [],
    allowed_time_windows: list = [],
) -> pd.DataFrame:
    subset_df = subset_df.copy().reset_index(drop=True)
    num_flights = len(subset_df)
    if num_flights == 0:
        return subset_df

    # Filter flights
    valid_flights_mask = filter_flights(
        subset_df,
        allowed_airlines,
        allowed_aircraft_types,
        allowed_stand_rwy_combinations,
        allowed_time_windows,
    )

    # Apply the mask to filter the DataFrame
    filtered_df = subset_df[valid_flights_mask].copy()
    filtered_out_df = subset_df[~valid_flights_mask].copy()

    # Assign default values for filtered-out flights
    filtered_out_df["Adjusted_Fuel_Consumption"] = filtered_out_df["F_i_norm"]
    filtered_out_df["Assigned_Tug"] = None

    # If no flights remain after filtering, return the filtered-out DataFrame
    if len(filtered_df) == 0:
        return filtered_out_df

    # Sort and prepare the filtered DataFrame
    filtered_df = filtered_df.sort_values("startMovement").reset_index(drop=True)
    flights = list(filtered_df.index)
    tugs = list(range(n_tugs))

    # Initialize the optimization model
    model = Model("AGPS_Tug_Assignment")
    model.setParam("OutputFlag", 0)

    # Decision variables
    x = model.addVars(flights, tugs, vtype=GRB.BINARY, name="x")
    z = model.addVars(flights, tugs, vtype=GRB.CONTINUOUS, lb=0.0, name="z")
    w = model.addVars(flights, vtype=GRB.BINARY, name="w")

    # Objective
    obj = quicksum(
        x[i, j] * filtered_df.loc[i, "F_i_agps"] for i in flights for j in tugs
    )
    obj += quicksum(z[i, j] for i in flights for j in tugs)
    obj += quicksum(w[i] * filtered_df.loc[i, "F_i_norm"] for i in flights)
    model.setObjective(obj, GRB.MINIMIZE)

    # Each flight must be assigned to one tug or go solo
    for i in flights:
        model.addConstr(quicksum(x[i, j] for j in tugs) + w[i] == 1, name=f"assign_{i}")

    # Conflict constraints
    for i_pos, i in enumerate(flights):
        for k_pos in range(i_pos + 1, len(flights)):
            k = flights[k_pos]

            if filtered_df.loc[k, "startMovement"] > filtered_df.loc[
                i, "startMovement"
            ] + pd.Timedelta(hours=3):
                break

            transit_distance = calculate_driving_distance(
                filtered_df.loc[i, "takeoffRunway"],
                filtered_df.loc[k, "stand_area"],
                distance_matrix,
            )
            transit_time_hours = transit_distance / agps_speed
            transit_time_td = pd.Timedelta(hours=transit_time_hours)
            required_start = (
                filtered_df.loc[i, "lineupTime"] + transit_time_td + buffer_time
            )

            if filtered_df.loc[k, "startMovement"] <= required_start:
                for j in tugs:
                    model.addConstr(
                        x[i, j] + x[k, j] <= 1, name=f"conflict_{i}_{k}_tug{j}"
                    )

    # Driving fuel consumption constraints
    for j in tugs:
        previous_position = None
        for i in flights:
            if previous_position is not None:
                distance = calculate_driving_distance(
                    previous_position, filtered_df.loc[i, "stand_area"], distance_matrix
                )
                driving_time = distance / agps_speed  # in hours
                tug_fuel = driving_time * agps_sfc
                model.addConstr(z[i, j] >= tug_fuel * x[i, j], name=f"fuel_{i}_{j}")
            previous_position = filtered_df.loc[i, "takeoffRunway"]

    # Solve the optimization model
    model.optimize()

    # Collect results
    assigned_tugs = []
    for i in flights:
        assigned_tug = None
        for j in tugs:
            if x[i, j].X > 0.5:
                assigned_tug = j
                break
        assigned_tugs.append(assigned_tug)

    filtered_df["Assigned_Tug"] = assigned_tugs
    filtered_df["Adjusted_Fuel_Consumption"] = [
        row.F_i_agps if tug is not None else row.F_i_norm
        for row, tug in zip(filtered_df.itertuples(index=False), assigned_tugs)
    ]

    # Combine filtered and filtered-out DataFrames
    valid_dfs = []

    for df in [filtered_df, filtered_out_df]:
        if isinstance(df, pd.DataFrame) and not df.empty:
            df_clean = df.dropna(axis=1, how="all")
            if not df_clean.empty:
                valid_dfs.append(df_clean)

    if valid_dfs:
        result_df = pd.concat(valid_dfs, ignore_index=True, sort=False)
    else:
        result_df = pd.DataFrame()
    return result_df


def run_optimization_for_multiDays_MILP(
    n_tugs: int,
    df_filtered: pd.DataFrame,
    num_workers: int = 4,
    distance_matrix: pd.DataFrame = DISTANCE_MATRIX,
    buffer_time: pd.Timedelta = DEFAULT_BUFFER_AGPS,
    agps_speed: float = DEFAULT_SPEED_AGPS,
    agps_sfc: float = DEFAULT_SFC_AGPS,
    allowed_airlines: list = [],
    allowed_aircraft_types: list = [],
    allowed_stand_rwy_combinations: list = [],
    allowed_time_windows: list = [],
) -> pd.DataFrame:
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                optimize_day_MILP_gurobi,
                # date,
                subset_df,
                n_tugs,
                distance_matrix,
                buffer_time,
                agps_speed,
                agps_sfc,
                allowed_airlines,
                allowed_aircraft_types,
                allowed_stand_rwy_combinations,
                allowed_time_windows,
            ): date
            for date, subset_df in df_filtered.groupby("date")
        }

        for future in tqdm(
            as_completed(futures), total=len(futures), desc=f"n_tugs = {n_tugs}"
        ):
            try:
                result = future.result()
                if not result.empty:
                    results.append(result)
            except Exception as e:
                print(f"Error processing a day's optimization: {e}")

    # Combine results into a single DataFrame for the current n_tugs
    if results:
        valid_results = []

        for df in results:
            if (
                isinstance(df, pd.DataFrame)
                and not df.empty
                and not df.isna().all().all()
            ):
                valid_results.append(df)

        if len(valid_results) > 0:
            df_results = pd.concat(valid_results, ignore_index=True)
        else:
            df_results = pd.DataFrame()
    else:
        df_results = pd.DataFrame()

    return df_results  # Return the final DataFrame


def get_drive_segments(
    df: pd.DataFrame, distance_matrix: pd.DataFrame = DISTANCE_MATRIX
) -> list:
    drive_segments = []

    # Prepare: drop NaNs early
    df = df.dropna(subset=["Assigned_Tug"]).copy()
    df = df.sort_values(["Assigned_Tug", "startMovement"]).reset_index(drop=True)

    # Map tug -> y-axis order
    unique_tugs = df["Assigned_Tug"].unique()
    tug_to_y = {tug: i for i, tug in enumerate(unique_tugs)}

    # Group by each tug
    grouped = df.groupby("Assigned_Tug")

    for tug, group in grouped:
        y = tug_to_y[tug]

        start_movements = group["startMovement"].values
        lineup_times = group["lineupTime"].values
        runways = group["takeoffRunway"].values
        stands = group["stand_area"].values

        # Process each flight and the next flight for the same tug
        for i in range(len(group) - 1):
            curr_end = lineup_times[i]
            curr_runway = runways[i]
            next_start = start_movements[i + 1]
            next_stand = stands[i + 1]

            # Only create a segment if next flight starts after the current ends
            if next_start > curr_end:
                distance_km = calculate_driving_distance(
                    curr_runway, next_stand, distance_matrix
                )
                drive_time_hr = distance_km / DEFAULT_SPEED_AGPS
                drive_time_td = timedelta(hours=drive_time_hr)

                drive_segments.append(
                    {
                        "tug": tug,
                        "start": curr_end,
                        "end": curr_end + drive_time_td,
                        "y": y,
                        "distance": distance_km,
                    }
                )

    return drive_segments


######
######
# def optimize_day_MILP(
#     date: pd.Timestamp,
#     subset_df: pd.DataFrame,
#     n_tugs: int,
#     distance_matrix: pd.DataFrame,
#     buffer_time: pd.Timedelta,
#     allowed_aircraft_types: list = [],
#     allowed_combinations: list = [],
# ) -> pd.DataFrame:
#     # Make a copy and reset the index.
#     subset_df = subset_df.copy().reset_index(drop=True)
#     flights = subset_df.index.tolist()
#     tugs = range(n_tugs)
#     num_flights = len(subset_df)

#     if num_flights == 0:
#         return subset_df  # Return empty DataFrame if no flights.

#     # Build a mask for flights that pass filtering.
#     valid_flights_mask = pd.Series([True] * len(subset_df))
#     if allowed_aircraft_types:
#         valid_flights_mask &= subset_df["typecode"].isin(allowed_aircraft_types)
#     if allowed_combinations:
#         valid_flights_mask &= subset_df.apply(
#             lambda row: (row["stand_area"], row["takeoffRunway"])
#             in allowed_combinations,
#             axis=1,
#         )

#     # Split into flights to optimize and those to remain unchanged.
#     filtered_df = subset_df[valid_flights_mask].copy()
#     filtered_out_df = subset_df[~valid_flights_mask].copy()

#     # For flights that are filtered out, assign default values.
#     filtered_out_df["Adjusted_Fuel_Consumption"] = filtered_out_df["F_i_norm"]
#     filtered_out_df["Assigned_Tug"] = None

#     # If there are no flights left to optimize, return the filtered-out DataFrame.
#     if len(filtered_df) == 0:
#         return filtered_out_df

#     flights = filtered_df.index.tolist()

#     # Create the MILP problem.
#     prob = LpProblem("Tug_Assignment_Problem", LpMinimize)

#     # Decision variables for tug assignment, extra fuel consumption (z), and taxiing without tug (w).
#     x = LpVariable.dicts("x", [(i, j) for i in flights for j in tugs], cat=LpBinary)
#     z = LpVariable.dicts(
#         "z", [(i, j) for i in flights for j in tugs], lowBound=0, cat=LpContinuous
#     )
#     w = LpVariable.dicts("w", flights, cat=LpBinary)

#     # Define the objective function.
#     prob += (
#         lpSum([x[i, j] * filtered_df.loc[i, "F_i_agps"] for i in flights for j in tugs])
#         + lpSum([z[i, j] for i in flights for j in tugs])
#         + lpSum([w[i] * filtered_df.loc[i, "F_i_norm"] for i in flights])
#     )

#     # Constraint 1: Each flight is either towed by one tug or taxis itself.
#     for i in flights:
#         prob += lpSum([x[i, j] for j in tugs]) + w[i] == 1

#     # Constraint 2: Prevent overlapping tug assignments on the same tug considering the buffer time.
#     # Refined to only consider flights that are relevant (ignore past flights and skip flights beyond a 3-hour window).
#     sorted_flight_indices = filtered_df.sort_values("startMovement").index.tolist()
#     for j in tugs:
#         for idx, i in enumerate(sorted_flight_indices):
#             t_i_start = filtered_df.loc[i, "startMovement"]
#             t_i_end = filtered_df.loc[i, "lineupTime"] + buffer_time

#             # Only consider flights coming after i.
#             for k in sorted_flight_indices[idx + 1 :]:
#                 t_k_start = filtered_df.loc[k, "startMovement"]
#                 # Break the loop if the flight is in the distant future (more than 2 hours ahead).
#                 if t_k_start > t_i_start + pd.Timedelta(hours=2):
#                     break
#                 t_k_end = filtered_df.loc[k, "lineupTime"] + buffer_time

#                 # Check if the time intervals overlap.
#                 if not (t_i_end <= t_k_start or t_k_end <= t_i_start):
#                     prob += x[i, j] + x[k, j] <= 1

#     # Constraint 3: Calculate tug fuel consumption (using a simple driving model).
#     sfc_tug = DEFAULT_SFC_AGPS  # kg/h
#     speed_tug = DEFAULT_SPEED_AGPS  # km/h
#     for j in tugs:
#         previous_position = None
#         for i in flights:
#             if previous_position is not None:
#                 distance = calculate_driving_distance(
#                     previous_position, filtered_df.loc[i, "stand_area"], distance_matrix
#                 )
#                 driving_time = distance / speed_tug
#                 fuel_consumption_tug = driving_time * sfc_tug
#                 prob += z[i, j] >= fuel_consumption_tug * x[i, j]
#             previous_position = filtered_df.loc[i, "takeoffRunway"]

#     # Solve the MILP problem.
#     prob.solve(PULP_CBC_CMD(msg=False))

#     # Retrieve tug assignment results.
#     assigned_tugs = []
#     for i in flights:
#         assigned_tug = None
#         for j in tugs:
#             if x[i, j].varValue is not None and x[i, j].varValue == 1:
#                 assigned_tug = j
#                 break
#         assigned_tugs.append(assigned_tug)

#     # Save results in the filtered DataFrame.
#     filtered_df["Assigned_Tug"] = pd.Series(assigned_tugs)
#     filtered_df["Adjusted_Fuel_Consumption"] = filtered_df.apply(
#         lambda row: row["F_i_norm"]
#         if pd.isna(row["Assigned_Tug"])
#         else row["F_i_agps"],
#         axis=1,
#     )

#     # Combine optimized and filtered-out flights.
#     valid_dfs = []
#     for df in [filtered_df, filtered_out_df]:
#         if not df.empty:
#             # Drop columns that are entirely NA or missing to avoid warnings in future pandas versions.
#             df_clean = df.dropna(axis=1, how="all")
#             if not df_clean.empty:
#                 valid_dfs.append(df_clean)

#     # Concatenate only if there are valid DataFrames to merge.
#     if valid_dfs:
#         result_df = pd.concat(valid_dfs, ignore_index=True, sort=False)
#     else:
#         result_df = pd.DataFrame()

#     return result_df
