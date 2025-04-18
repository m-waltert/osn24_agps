from pulp import (
    LpProblem,
    LpMinimize,
    LpVariable,
    LpBinary,
    LpContinuous,
    lpSum,
    PULP_CBC_CMD,
)

from gurobipy import Model, GRB, quicksum

import pandas as pd
import numpy as np
from scipy.optimize import linprog


import datetime
from agps_config import DEFAULT_SFC_AGPS, DEFAULT_SPEED_AGPS

# Driving distances from Runway to Stand Areas in kilometers [km]
# All runways to unknwon: 2.8km (which is equal to the average of the distances to the other stand areas)
distance_matrix_LSZH = pd.DataFrame(
    data=[  # A_N    AB_C    B_S     C,      D,      E,      F,      G,      H,      I,      T,      GAC     unknown
        [4.4, 3.8, 3.7, 3.5, 3.3, 5.3, 4.6, 2.8, 4.4, 4.7, 3.3, 5.0, 2.8],  # RWY 10
        [3.6, 3.6, 3.8, 4.4, 4.2, 2.1, 3.7, 5.1, 3.6, 3.9, 4.6, 4.1, 2.8],  # RWY 16
        [1.3, 1.5, 1.5, 2.0, 1.8, 2.0, 0.6, 2.7, 1.2, 1.5, 2.2, 0.4, 2.8],  # RWY 28
        [2.4, 2.5, 2.7, 3.3, 3.0, 1.1, 1.8, 3.8, 2.4, 2.7, 3.3, 1.6, 2.8],  # RWY 32
        [2.2, 1.7, 1.4, 1.3, 1.0, 3.0, 2.4, 0.5, 2.2, 2.4, 1.0, 2.7, 2.8],  # RWY 34
    ],
    index=["10", "16", "28", "32", "34"],  # Takeoff runways as rows
    columns=[
        "A North",
        "AB Courtyard",
        "B South",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "T",
        "GAC",
        "unknown",
    ],  # Stand Areas as columns
)


def calculate_driving_distance(runway: str, stand_area: str, distance_matrix) -> float:
    """
    Retrieve distance between runway and stand area from the distance matrix.
    If the combination does not exist, return a default value.
    """
    try:
        return distance_matrix.loc[runway, stand_area]
    except KeyError:
        return 2.8  # Default distance if not found


def optimize_day_MILP_gurobi(
    date: pd.Timestamp,
    subset_df: pd.DataFrame,
    n_tugs: int,
    distance_matrix: pd.DataFrame,
    buffer_time: pd.Timedelta,
    allowed_aircraft_types: list = [],
    allowed_combinations: list = [],
) -> pd.DataFrame:
    subset_df = subset_df.copy().reset_index(drop=True)
    num_flights = len(subset_df)
    if num_flights == 0:
        return subset_df

    # Filter flights
    valid_flights_mask = pd.Series([True] * len(subset_df))
    if allowed_aircraft_types:
        valid_flights_mask &= subset_df["typecode"].isin(allowed_aircraft_types)
    if allowed_combinations:
        valid_flights_mask &= subset_df.apply(
            lambda row: (row["stand_area"], row["takeoffRunway"])
            in allowed_combinations,
            axis=1,
        )

    filtered_df = subset_df[valid_flights_mask].copy()
    filtered_out_df = subset_df[~valid_flights_mask].copy()

    filtered_out_df["Adjusted_Fuel_Consumption"] = filtered_out_df["F_i_norm"]
    filtered_out_df["Assigned_Tug"] = None

    if len(filtered_df) == 0:
        return filtered_out_df

    filtered_df = filtered_df.sort_values("startMovement").reset_index(drop=True)
    flights = list(filtered_df.index)
    tugs = list(range(n_tugs))

    speed_tug = DEFAULT_SPEED_AGPS  # in km/h
    sfc_tug = DEFAULT_SFC_AGPS  # in kg/h

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
            transit_time_hours = transit_distance / speed_tug
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
                driving_time = distance / speed_tug  # in hours
                tug_fuel = driving_time * sfc_tug
                model.addConstr(z[i, j] >= tug_fuel * x[i, j], name=f"fuel_{i}_{j}")
            previous_position = filtered_df.loc[i, "takeoffRunway"]

    # Solve
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

    valid_dfs = []
    for df in [filtered_df, filtered_out_df]:
        if not df.empty and not df.isna().all(axis=None):
            valid_dfs.append(df)

    if valid_dfs:
        result_df = pd.concat(valid_dfs, ignore_index=True, sort=False)
    else:
        result_df = pd.DataFrame()
    return result_df


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
