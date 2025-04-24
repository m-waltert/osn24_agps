from agps_config import (
    AIRCRAFT_INFO,
    DEFAULT_STARTUP_TIME,
    DEFAULT_WARMUP_TIME,
    DF_APU,
    OLD_AIRCRAFT,
    AC2CONSIDER,
    DF_MISSING_ICAO24,
    get_Stands_LSZH,
)
import traffic
from traffic.core import Traffic, Flight
from traffic.data import airports
import numpy as np
from datetime import timedelta
from shapely.geometry import base
from shapely.geometry import LineString
import pandas as pd
import datetime
import os
from openap import prop
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from cartes.crs import EuroPP
import cartopy.crs as ccrs
from shapely.geometry import Point

data_proj = ccrs.PlateCarree()


def get_lon_lat_by_runway(df_runways, runway_name):
    """
    Retrieves the longitude and latitude coordinates for a specified runway.

    This function searches a DataFrame containing runway information for a row
    matching the provided `runway_name`. If found, it extracts and returns the
    longitude and latitude of that runway; otherwise, it returns `None` values.
    """

    row = df_runways[df_runways["name"] == runway_name]
    if not row.empty:
        lat = row.iloc[0]["latitude"]
        lon = row.iloc[0]["longitude"]
        return lon, lat
    else:
        return None, None


def extend_point(p1, p2, distance):
    """
    Extends a point `p1` away from a second point `p2` by a specified distance.

    This function calculates the direction vector from `p1` to `p2`, normalizes it,
    and then extends `p1` in the opposite direction of `p2` by the given `distance`.
    This is useful for creating extended points along or opposite to a line segment.
    """

    # Calculate direction vector
    direction = np.array(p2) - np.array(p1)
    direction = direction / np.linalg.norm(direction)  # Normalize the vector
    # Extend the point
    new_point = np.array(p1) - direction * distance
    return tuple(new_point)


def find_opposite_runway(runway: str) -> str:
    """
    Determines the opposite runway identifier based on a given runway number and position.

    The opposite runway is typically located 180 degrees from the original runway,
    with an identifier that is 18 units different from the original. The function
    also handles the left ('L'), right ('R'), and center ('C') suffixes to ensure the
    correct opposite is returned.

    Parameters:
    -----------
    runway : str
        The original runway identifier, which may include a suffix ('L', 'R', or 'C')
        to indicate position (e.g., '09L', '27R').

    Returns:
    --------
    str
        The opposite runway identifier, adjusted by 180 degrees and including the
        appropriate suffix if applicable.

    Notes:
    ------
    - Runway numbers are represented in a range of 01 to 36. If the computed opposite
      number is 0, it wraps around to 36.
    - Left ('L') and right ('R') suffixes are swapped for the opposite runway, while
      the center ('C') suffix remains unchanged.

    Example:
    --------
    >>> find_opposite_runway("09L")
    '27R'

    >>> find_opposite_runway("36")
    '18'
    """

    # Extract the number part of the runway
    number_part = int(runway[:-1] if runway[-1] in "LRC" else runway)

    # Compute the opposite runway number by adding/subtracting 18
    opposite_number = (number_part + 18) % 36
    if opposite_number == 0:
        opposite_number = 36

    # Format the number with leading zero if it's less than 10
    opposite_number_str = f"{opposite_number:02d}"

    # Handle the suffix if present
    if runway[-1] == "L":
        suffix = "R"
    elif runway[-1] == "R":
        suffix = "L"
    elif runway[-1] == "C":
        suffix = "C"
    else:
        suffix = ""

    # Combine the opposite number with the suffix
    return f"{opposite_number_str}{suffix}"


def get_Box_around_Rwy(
    Rwy: str, airport_str: str, extension_distance=0.002, extension_width=0.0004
):
    """
    Generates a rectangular bounding box around a specified runway by extending its length
    and width, based on given parameters. This bounding box can be useful for spatial analysis
    or for identifying areas around the runway.
    """

    # Find runway opposite to rwy
    oppositeRwy = find_opposite_runway(Rwy)

    # Get runway data from traffic library
    runways = airports[airport_str].runways.data

    # Define the coordinates of the two points
    lon1, lat1 = get_lon_lat_by_runway(runways, Rwy)
    lon2, lat2 = get_lon_lat_by_runway(runways, oppositeRwy)

    # Define the coordinates of the two points
    point1 = (lon1, lat1)
    point2 = (lon2, lat2)

    # Create a LineString object
    rwy = LineString([point1, point2])

    # Extend both ends of the line
    rwy_box = LineString(
        [
            extend_point(point2, point1, extension_distance),
            extend_point(point1, point2, extension_distance),
        ]
    )

    # Extend width by buffering
    rwy_box = rwy_box.buffer(distance=extension_width, cap_style="square")
    return rwy_box


def takeoff_detection(flight: Flight, airport_str="LSZH") -> str:
    """
    Detects takeoff events from a given aircraft trajectory and updates its trajectory data.

    This function identifies whether a provided flight trajectory corresponds to a takeoff event
    at a specified airport. If detected, the function determines the runway used and the lineup time,
    which are then added as new attributes to the trajectory data.

    Args:
        flight (Flight): A traffic.core.Flight object containing trajectory data, including positions
                         and timestamps.
        airport_str (str, optional): The ICAO code of the airport to detect takeoff from. Defaults to 'LSZH'.

    Returns:
        Flight: The same Flight object with additional columns in its data:
            - 'lineupTime' (pd.Timestamp or NaN): The timestamp of lineup on the runway, or NaN if no takeoff is detected.
            - 'takeoffRunway' (str): The runway used for takeoff, or an empty string if no runway is detected.
            - 'isTakeoff' (bool): A boolean flag indicating whether a takeoff was detected.

    Notes:
        - The function uses a track-based method to detect takeoff events.
        - The detection process involves generating a geometry box around the detected runway.
        - The trajectory is clipped to this geometry box for improved accuracy in takeoff detection.
        - This function modifies the 'flight.data' DataFrame by adding new columns: 'lineupTime', 'takeoffRunway', and 'isTakeoff'.

    Example:
        >>> updated_flight = takeoff_detection(flight, airport_str='LSZH')
    """

    lineupTime = np.nan
    isTakeoff = False
    takeoffRunway = ""

    if takeoff := flight.takeoff(airport_str, method="track_based").next():
        takeoffRunway = takeoff.runway_max

        if takeoffRunway is not None:
            isTakeoff = True

            # Get runway geometry (box around runway)
            rwy_box = get_Box_around_Rwy(takeoffRunway, airport_str)

            clipped = flight.clip(rwy_box)

            if clipped is not None and not clipped.data.empty:
                lineupTime = clipped.start
        else:
            takeoffRunway = ""

    flight.data = flight.data.copy()
    flight.data.loc[:, "lineupTime"] = lineupTime
    flight.data.loc[:, "takeoffRunway"] = takeoffRunway
    flight.data.loc[:, "isTakeoff"] = isTakeoff

    return flight


def apron_events(
    flight: Flight, standAreas: list, airport_str="LSZH", gs_threshold=1
) -> Flight:
    lineupTime = pd.Timestamp(flight.data.lineupTime.iloc[0])
    taxiDuration = np.nan
    taxiDistance = np.nan
    startPushback = np.nan
    pushbackDuration = np.nan
    startTaxi = np.nan
    isPushback = False
    parking_position = None
    """
    Detects and annotates apron-level events including pushback, taxi, and parking position for a given flight.

    This function analyzes the ground movement of a flight before takeoff to identify key events such as pushback, 
    taxi start, pushback duration, taxi duration, taxi distance, and parking position. It uses spatial and temporal 
    filtering techniques to detect these events and applies only to flights that are marked as takeoffs.

    Args:
        flight (Flight): A traffic.core.Flight object containing trajectory data, including positions and timestamps.
        standAreas (list): A list of shapely geometries representing nose-in, push-out stand areas at the airport.
        airport_str (str, optional): The ICAO code of the airport for which parking position information is retrieved. 
                                     Defaults to "LSZH" (Zurich Airport).
        gs_threshold (float, optional): Groundspeed threshold (in knots or m/s) used to determine when the aircraft 
                                        is considered moving. Defaults to 1.

    Returns:
        Flight: The same Flight object with the following additional columns in its `flight.data` DataFrame:
            - 'isPushback' (bool): True if a pushback maneuver was detected, False otherwise.
            - 'startPushback' (pd.Timestamp or NaT): Start time of the pushback maneuver.
            - 'startTaxi' (pd.Timestamp or NaT): Start time of the taxi phase.
            - 'pushbackDuration' (pd.Timedelta or NaT): Duration of the pushback maneuver.
            - 'taxiDuration' (pd.Timedelta or NaN): Duration of the taxi phase until lineup.
            - 'taxiDistance' (float or NaN): Distance covered during taxi, computed from cumulative distance.
            - 'parking_position' (str or None): Detected parking position, if available.

    Notes:
        - Pushback is detected if the flight trajectory intersects with any predefined stand area geometry for 
          a duration of at least 20 seconds.
        - The function applies a rolling median filter to smooth groundspeed values before detecting motion.
        - Trajectories are filtered to exclude data points within 0.03 NM (approximately 50 meters) of the initial position,
          which helps eliminate noisy data when the aircraft is stationary.
        - The function groups trajectory points into motion segments using a timestamp difference of more than 1 minute.
        - If pushback is not detected, the start of taxi is estimated from the first point where groundspeed exceeds the threshold.
        - Parking position is estimated based on trajectory segments closest to the apron departure time.
        - The function adds several new columns to the `flight.data` DataFrame, which should be handled appropriately when analyzing results.

    Example:
        >>> updated_flight = apron_events(flight, standAreas, airport_str='LSZH', gs_threshold=1)
    """

    # Apply pushback and taxi detection only to takeoffs
    if flight.isTakeoff_max:
        # Get taxi-part of trajectory
        # Calculate the position of the flight from its initial position (i.e. the stand)
        if len(flight.first(1).data) > 1:
            flight = flight.resample("1s")
        first_position = flight.at(flight.start)
        flight = flight.distance(other=first_position)  # add column distance

        # only consider the part of the trajectory which is 0.03 away from the initial position (to avoid noisy data on the stands)
        taxi = flight.query("distance > 0.03")

        # Get parts of flight where it is moving
        if taxi is not None:
            # moving = taxi.data[taxi.data["compute_gs"] > gs_threshold].copy()

            moving = taxi.data.copy()
            moving["compute_gs"] = (
                moving["compute_gs"].rolling(window=21, center=True).median()
            )
            moving = moving[moving.compute_gs > gs_threshold]
        else:
            moving = flight.data.copy()
            taxi = flight

        # Pushback Detection
        for i, standArea in enumerate(standAreas):
            clipped_traj = taxi.clip(standArea)

            # Check whether traj is inside stand_area
            if (
                (clipped_traj is None)
                or (clipped_traj.data.empty)
                or clipped_traj.duration < pd.Timedelta(seconds=20)
            ):
                continue
            else:
                isPushback = True
                break

        if isPushback and (clipped_traj is not None):
            # Get time when the flight leaves the stand area
            leaveStandTime = clipped_traj.stop

            # Identify consecutive groups where compute_gs is above threshold -> groups of "moving"
            moving["group"] = (
                (moving["timestamp"].diff() > pd.Timedelta(minutes=1))
                .fillna(False)
                .astype("int")
                .cumsum()
            )

            # Find the group where leaveStandTime falls within the timestamp range
            group = moving[
                (moving["timestamp"] <= leaveStandTime)
                & (moving["timestamp"].shift(-1) >= leaveStandTime)
            ]["group"]

            if not group.empty:
                group_id = group.iloc[0]

                # Extract the start and stop timestamps for that group
                group_data = moving[moving["group"] == group_id]
                startPushback = pd.Timestamp(group_data["timestamp"].min())
                startTaxi = pd.Timestamp(group_data["timestamp"].max())
            else:
                startPushback = leaveStandTime
                startTaxi = leaveStandTime

            # Pushback duration
            pushbackDuration = startTaxi - startPushback

            # Check if taxiDuration is less than 20 seconds. This can happen if the ground coverage is not perfect and parts of the traj are missing.
            # if (lineupTime - startTaxi) < timedelta(seconds=0):
            if (lineupTime - startTaxi) < pd.Timedelta(seconds=20):
                startTaxi = leaveStandTime
                isPushback = False
                startPushback = np.nan
                pushbackDuration = np.nan

            # Determine Parking Positions of Flight
            parking_segments = flight.parking_position(airport_str)

            # Make sure that the parking position is the segment closest to leaveStandTime
            if parking_segments is not None:
                timedeltas = []

                for segment in parking_segments:
                    timedeltas.append(abs(segment.stop - leaveStandTime))

                # Check whether the list contains elements:
                if timedeltas:
                    min_index = timedeltas.index(min(timedeltas))
                    parking_position = (
                        parking_segments[min_index].data.iloc[0].parking_position
                    )

        # If takeoff is not pushback, check for stand
        else:
            # Select parking position segment with longest duration
            try:
                if flight.parking_position is not None:
                    parking = flight.parking_position(airport_str).max()
                else:
                    parking = None
            except:
                parking = None

            # Determine start of taxi (moment when the aircraft starts moving)
            if not moving.empty:
                if moving.timestamp.iloc[0] is not None:
                    startTaxi = moving.timestamp.iloc[0]

            # Determine Parking Position of Flight (if there is one)
            if (parking is not None) and (parking.duration > pd.Timedelta(seconds=30)):
                parking_position = parking.parking_position_max

        # Calculate taxiDuration
        if (
            (startTaxi is not np.nan)
            & (startTaxi is not None)
            & (lineupTime is not np.nan)
            & (lineupTime is not None)
        ):
            if (
                lineupTime - startTaxi >= pd.Timedelta(seconds=2)
            ):  # Do not remove the 2 seconds --> for shorter lineup-taxi times, it produces an error
                taxiDuration = lineupTime - startTaxi
                taxiDistance = (
                    flight.between(startTaxi, lineupTime).data.cumdist.iloc[-1]
                    - flight.between(startTaxi, lineupTime).data.cumdist.iloc[0]
                )
            else:
                taxiDuration = pd.Timedelta(seconds=0)
                taxiDistance = 0

        # # Calculate taxiDistance
        # if taxiDuration is not np.nan:
        #     taxi = flight.between(startTaxi, lineupTime)

        #     if taxi is not None:
        #         taxiDistance = taxi.data.cumdist.iloc[-1] - taxi.data.cumdist.iloc[0]

    # Write data to traj
    flight.data = flight.data.copy()
    flight.data.loc[:, "isPushback"] = isPushback
    flight.data.loc[:, "startPushback"] = startPushback
    flight.data.loc[:, "startTaxi"] = startTaxi
    flight.data.loc[:, "pushbackDuration"] = pushbackDuration
    flight.data.loc[:, "taxiDuration"] = taxiDuration
    flight.data.loc[:, "taxiDistance"] = taxiDistance
    flight.data.loc[:, "parking_position"] = parking_position

    return flight


def taxi_dist_dur(flight: Flight) -> Flight:
    """
    Computes taxi distance and duration for a flight trajectory.

    This function calculates the taxi distance and duration from the start of taxi to lineup time
    for a given flight trajectory. It updates the flight data with these computed values.

    Parameters:
        flight (Flight): A traffic.core.Flight object containing trajectory data,
                         with computed `startTaxi` and `lineupTime`.

    Returns:
        Flight: The same Flight object, with additional columns added to `flight.data`:
            - `taxiDuration`: Duration of the taxi phase.
            - `taxiDistance`: Distance covered during taxiing.

    Notes:
        - Taxi distance and duration are calculated only if both `startTaxi` and `lineupTime` are defined.
        - If the taxi duration is less than 2 seconds, it is set to 0.
    """

    startTaxi = flight.data.startTaxi.iloc[0]
    lineupTime = flight.data.lineupTime.iloc[0]
    taxiDuration = np.nan
    taxiDistance = np.nan

    # Apply pushback and taxi detection only to takeoffs
    if flight.isTakeoff_max:
        if (
            (startTaxi is not np.nan)
            & (startTaxi is not None)
            & (startTaxi is not pd.NA)
            & (lineupTime is not np.nan)
            & (lineupTime is not None)
            & (lineupTime is not pd.NA)
        ):
            if lineupTime - startTaxi >= pd.Timedelta(seconds=2):
                taxiDuration = lineupTime - startTaxi
                taxi_cumdist = flight.between(startTaxi, lineupTime).data.cumdist

                if taxi_cumdist is not None:
                    taxiDistance = taxi_cumdist.iloc[-1] - taxi_cumdist.iloc[0]
                else:
                    taxiDistance = 0
            else:
                taxiDuration = pd.Timedelta(seconds=0)
                taxiDistance = 0

    flight.data = flight.data.copy()
    flight.data.loc[:, "taxiDuration"] = taxiDuration
    flight.data.loc[:, "taxiDistance"] = taxiDistance

    return flight


def gs_cumdist_diff(flight: Flight) -> Flight:
    startTaxi = flight.data.startTaxi.iloc[0]
    lineupTime = flight.data.lineupTime.iloc[0]
    min_compute_gs_diff = np.nan
    max_compute_gs_diff = np.nan
    max_cumdist_diff = np.nan
    first_altitude = np.nan
    """
    Computes diagnostic metrics for a flight trajectory during taxi, to help identify potential data issues
    or landings misclassified as takeoffs.

    This function calculates:
    - The maximum altitude reached during the first 10 seconds after takeoff.
    - The minimum and maximum differences in groundspeed (`compute_gs.diff()`).
    - The maximum difference in cumulative distance (`cumdist.diff()`).

    These values are useful for post-processing and filtering, particularly for identifying
    cases where landing trajectories might be misinterpreted as takeoffs.

    Parameters:
        flight (Flight): A `traffic.core.Flight` object containing trajectory data,
                         with computed `startTaxi`, `lineupTime`, and `compute_gs`.

    Returns:
        Flight: The same Flight object, with additional diagnostic columns added to `flight.data`:
            - `first_altitude`: Maximum observed altitude in the first 10 seconds of the flight.
            - `min_compute_gs_diff`: Minimum difference in groundspeed during taxi.
            - `max_compute_gs_diff`: Maximum difference in groundspeed during taxi.
            - `max_cumdist_diff`: Maximum difference in cumulative distance during taxi.

    Notes:
        - Metrics are only computed for takeoff flights (`flight.isTakeoff_max` is True).
        - If the taxi interval is not well-defined or less than 2 seconds, fallback logic uses data before lineup time.
        - These indicators can help detect abnormal taxi profiles or wrongly detected takeoffs.
    """

    if flight.isTakeoff_max:
        # Get max observed altitude during first 10 seconds of flight --> Used to identify landings wrongly classified as takeoffs
        first_altitude = flight.first(seconds=10).data.altitude.max()

        # Calculate min and max compute_gs.diff() for part on the ground
        if (startTaxi is not np.nan) | (lineupTime is not np.nan):
            if lineupTime - startTaxi >= pd.Timedelta(seconds=2):
                compute_gs = flight.between(startTaxi, lineupTime).data["compute_gs"]
                cumdist = flight.between(startTaxi, lineupTime).data["cumdist"]
                if compute_gs is not None:
                    min_compute_gs_diff = compute_gs.diff().min()
                    max_compute_gs_diff = compute_gs.diff().max()
                    max_cumdist_diff = cumdist.diff().max()

        else:
            compute_gs = flight.before(lineupTime).data["compute_gs"]
            cumdist = flight.before(lineupTime).data["cumdist"]
            if compute_gs is not None:
                min_compute_gs_diff = compute_gs.diff().min()
                max_compute_gs_diff = compute_gs.diff().max()
                max_cumdist_diff = cumdist.diff().max()

    flight.data = flight.data.copy()
    flight.data.loc[:, "first_altitude"] = first_altitude
    flight.data.loc[:, "min_compute_gs_diff"] = min_compute_gs_diff
    flight.data.loc[:, "max_compute_gs_diff"] = max_compute_gs_diff
    flight.data.loc[:, "max_cumdist_diff"] = max_cumdist_diff

    return flight


def get_stand_area(flight: Flight, standAreas) -> Flight:
    # Default value for area
    area = "unknown"

    if flight.isTakeoff_max:
        # Get the first position of the flight
        first_position = flight.first(5)

        if first_position is not None:
            lon = first_position.data.longitude.mean()
            lat = first_position.data.latitude.mean()

            if lat is not None and lon is not None:
                first_point = Point(lon, lat)

            for stand_name, polygon in standAreas.items():
                if polygon.contains(first_point):
                    area = stand_name
                    break

    flight.data = flight.data.copy()
    flight.data.loc[:, "stand_area"] = area

    return flight


def add_known_missing_icao24(df_movements: pd.DataFrame) -> pd.DataFrame:
    """
    Enhances the aircraft movement DataFrame by merging with a dataset of missing ICAO24 codes to fill in
    missing aircraft type and typecode information. This function helps in completing aircraft information
    based on known ICAO24 codes and handles specific column renaming and replacement tasks.

    Parameters:
    -----------
    df_movements : pd.DataFrame
        DataFrame containing aircraft movements, with columns including 'icao24', 'typecode', and 'icaoaircrafttype'.

    Returns:
    --------
    pd.DataFrame
        Updated DataFrame with merged and completed aircraft type and typecode information based on known missing ICAO24 entries.

    Process:
    --------
    - Merges `df_movements` with `DF_MISSING_ICAO24` on the 'icao24' column using a left join to pull in missing
      'typecode' and 'icaoaircrafttype' values where available.
    - Uses the `combine_first` method to prioritize original `df_movements` typecode values where they exist,
      filling in with the merged values if not.
    - Updates `icaoaircrafttype` based on available data in the merged columns.
    - Drops redundant columns resulting from the merge and renames the updated columns to match the original names.
    - Replaces instances of 'C68A' in the 'icaoaircrafttype' column with 'L2J' for standardized classification.

    Notes:
    ------
    - Assumes `DF_MISSING_ICAO24` is a DataFrame with columns 'icao24', 'typecode', and 'icaoaircrafttype'.
    - This function is useful in cases where certain `icao24` entries may be incomplete in the original dataset
      and a secondary dataset can help fill in the gaps.

    Example:
    --------
    df_enhanced = add_known_missing_icao24(df_movements)
    """

    # Merge df_movements with missing icao24 df
    df_movements = df_movements.merge(
        DF_MISSING_ICAO24[["icao24", "typecode", "icaoaircrafttype"]],
        on="icao24",
        how="left",
    )

    df_movements["typecode_x"] = df_movements["typecode_x"].combine_first(
        df_movements["typecode_y"]
    )
    # df_movements['icaoaircrafttype_x'] = df_movements['icaoaircrafttype_x'].combine_first(df_movements['icaoaircrafttype_y'])
    df_movements.loc[
        df_movements["icaoaircrafttype_y"].notna(), "icaoaircrafttype_x"
    ] = df_movements["icaoaircrafttype_y"]
    df_movements.drop(columns=["typecode_y", "icaoaircrafttype_y"], inplace=True)

    # Rename specific columns
    df_movements = df_movements.rename(
        columns={"typecode_x": "typecode", "icaoaircrafttype_x": "icaoaircrafttype"}
    )

    # Replace 'C68A' with 'L2J' in the 'icaoaircrafttype' column
    df_movements.loc[df_movements["icaoaircrafttype"] == "C68A", "icaoaircrafttype"] = (
        "L2J"
    )

    return df_movements


def remove_helos_and_outliers(df_movements: pd.DataFrame) -> pd.DataFrame:
    """
    Filters out helicopter movements and outliers from an aircraft movements DataFrame based on
    specific criteria to ensure data quality for further analysis.

    This function removes rows where:
    - The aircraft type is classified as a helicopter (indicated by an 'H' in the 'icaoaircrafttype' column).
    - Taxi distance or lineup time values are missing (NaN).
    - Taxi duration is less than zero, which may indicate erroneous data.

    Parameters:
    -----------
    df_movements : pd.DataFrame
        DataFrame containing aircraft movement data with columns such as 'icaoaircrafttype',
        'taxiDistance', 'lineupTime', and 'taxiDuration'.

    Returns:
    --------
    pd.DataFrame
        Cleaned DataFrame with helicopter movements and outliers removed.

    Filtering Criteria:
    -------------------
    - Helicopters are identified by checking if the 'icaoaircrafttype' column contains an 'H'.
    - Movements with NaN values in 'taxiDistance' and 'lineupTime' columns are removed.
    - Movements with negative taxi durations (indicating possible data errors) are filtered out.

    Example:
    --------
    df_filtered = remove_helos_and_outliers(df_movements)
    """

    # Remove takeoffs on runway 14
    df_movements = df_movements[~df_movements["takeoffRunway"].str.contains("14")]

    # Remove flights with an initial altitude (first 10 seconds) larger than 2000ft -> Landings wrongly classified as takeoffs.
    # We keep rows where first_altitude is NaN
    df_movements = df_movements[
        (
            (df_movements["first_altitude"] < 2000)
            | df_movements["first_altitude"].isna()
        )
    ]

    # Remove rows where 'icaoaircrafttype' contains 'H' anywhere in the string -> Remove helicopters
    df_movements = df_movements[
        ~df_movements["icaoaircrafttype"].str.contains("H", case=False, na=False)
    ]

    # Remove flights which have taxi distances of NaN
    df_movements = df_movements[~df_movements["taxiDistance"].isna()]

    # Remove flights which have lineup times of NaN
    df_movements = df_movements[~df_movements["lineupTime"].isna()]

    # Remove flights which have taxiDurations of 0 or less
    df_movements = df_movements[~(df_movements["taxiDuration"] <= pd.Timedelta(0))]

    # Remove flights whose max_compute_gs_diff is greater than 200 and max_cumdist_diff ist creater than 0.1
    df_movements = df_movements[
        (df_movements["max_compute_gs_diff"] < 200)
        & (df_movements["min_compute_gs_diff"] > -200)
        & (df_movements["max_cumdist_diff"] < 0.1)
    ]

    return df_movements


def get_df_movements(df) -> pd.DataFrame:
    """
    Filters helicopter movements and erroneous data points from an aircraft movements DataFrame.

    This function applies multiple filtering criteria to clean the input DataFrame by removing:
    - Helicopter movements identified via 'icaoaircrafttype' containing 'H'.
    - Data points with missing or invalid entries in critical columns ('taxiDistance', 'lineupTime', 'taxiDuration').
    - Movements with unrealistic taxi durations (e.g., zero or negative durations).
    - Landings mistakenly classified as takeoffs based on initial altitude thresholds.
    - Movements with extreme values in computed ground speed and cumulative distance differences.
    - Takeoffs from a specific runway (14), which may be outside the scope of analysis.

    Parameters:
    -----------
    df_movements : pd.DataFrame
        A DataFrame containing aircraft movement data with the following relevant columns:
        - 'icaoaircrafttype' (str): Aircraft type code.
        - 'takeoffRunway' (str): The runway used for takeoff.
        - 'first_altitude' (float): Maximum altitude reached during the first 10 seconds of flight.
        - 'taxiDistance' (float): The total taxi distance covered.
        - 'lineupTime' (Timestamp): The timestamp indicating lineup for takeoff.
        - 'taxiDuration' (Timedelta): The total time spent taxiing.
        - 'max_compute_gs_diff' (float): Maximum difference in computed groundspeed.
        - 'min_compute_gs_diff' (float): Minimum difference in computed groundspeed.
        - 'max_cumdist_diff' (float): Maximum difference in cumulative distance.

    Returns:
    --------
    pd.DataFrame
        A cleaned DataFrame with invalid rows removed according to the filtering criteria.

    Filtering Criteria:
    -------------------
    1. Helicopter Removal:
       - Excludes rows where 'icaoaircrafttype' contains 'H' (case-insensitive).
    2. Takeoff Runway Filtering:
       - Removes takeoffs from runway 14.
    3. Initial Altitude Filtering:
       - Excludes rows where 'first_altitude' > 2000 ft, allowing NaN values to remain.
    4. Missing Data Removal:
       - Excludes rows with NaN values in 'taxiDistance' and 'lineupTime'.
    5. Taxi Duration Filtering:
       - Excludes rows where 'taxiDuration' is zero or negative.
    6. Ground Speed and Distance Filtering:
       - Filters out rows where 'max_compute_gs_diff' > 200 or 'min_compute_gs_diff' < -200.
       - Filters out rows where 'max_cumdist_diff' >= 0.1.

    Example:
    --------
    >>> df_filtered = remove_helos_and_outliers(df_movements)
    """

    # Group by 'flight_id'
    grouped = df.query('takeoffRunway != ""').groupby("flight_id")

    # Create a new DataFrame df_movements to store the summarized data
    df_movements = pd.DataFrame()

    # Extract the required information
    df_movements["flight_id"] = grouped["flight_id"].first()
    df_movements["icao24"] = grouped["icao24"].first()
    df_movements["callsign"] = grouped["callsign"].first()
    df_movements["isTakeoff"] = grouped["isTakeoff"].first()
    df_movements["isPushback"] = grouped["isPushback"].first()
    df_movements["startPushback"] = grouped["startPushback"].first()
    df_movements["startTaxi"] = grouped["startTaxi"].first()
    df_movements["lineupTime"] = grouped["lineupTime"].first()
    df_movements["taxiDuration"] = grouped["taxiDuration"].first()
    df_movements["taxiDistance"] = grouped["taxiDistance"].first()
    df_movements["takeoffRunway"] = grouped["takeoffRunway"].first()
    df_movements["parking_position"] = grouped["parking_position"].first()
    df_movements["stand_area"] = grouped["stand_area"].first()
    df_movements["typecode"] = grouped["typecode"].first()
    df_movements["icaoaircrafttype"] = grouped["icaoaircrafttype"].first()
    df_movements["min_compute_gs_diff"] = grouped["min_compute_gs_diff"].first()
    df_movements["max_compute_gs_diff"] = grouped["max_compute_gs_diff"].first()
    df_movements["max_cumdist_diff"] = grouped["max_cumdist_diff"].first()
    df_movements["first_altitude"] = grouped["first_altitude"].first()

    # Reset index to get a clean DataFrame
    df_movements = df_movements.reset_index(drop=True)

    return df_movements


def normalTaxiFuel_df(
    df_movements: pd.DataFrame,
    startupTime=DEFAULT_STARTUP_TIME,
    warmupTime=DEFAULT_WARMUP_TIME,
    colNames=["MESengine", "MESapu", "normTAXIengine"],
) -> pd.DataFrame:
    """
    Calculates and adds columns for normal taxi fuel consumption to the aircraft movements DataFrame.

    This function computes fuel consumption for the main engine startup (MES), auxiliary power unit (APU),
    and normal taxiing phases, based on specified startup and warmup times. The resulting DataFrame will
    include new columns for each fuel consumption calculation, labeled according to the `colNames` parameter.

    Parameters:
    -----------
    df_movements : pd.DataFrame
        DataFrame containing aircraft movement data with relevant information for fuel calculations.

    startupTime : float, optional
        Time in seconds allocated for engine startup (default is `DEFAULT_STARTUP_TIME`).

    warmupTime : float, optional
        Time in seconds for engine warmup (default is `DEFAULT_WARMUP_TIME`).

    colNames : list of str, optional
        List containing the column names for storing calculated fuel consumption values.
        Defaults to `['MESengine', 'MESapu', 'normTAXIengine']` for main engine startup, APU,
        and taxi engine fuel consumption respectively.

    Returns:
    --------
    pd.DataFrame
        Updated DataFrame with added columns for fuel consumption:
        - `colNames[0]`: Fuel consumed during main engine startup.
        - `colNames[1]`: Fuel consumed by the APU during startup and warmup phases.
        - `colNames[2]`: Fuel consumed during the normal taxiing phase.

    Notes:
    ------
    - This function depends on external functions `fuelMESengine_df`, `fuelMESapu_df`, and `fuelTaxiEngine_df`
      to perform each fuel consumption calculation and update `df_movements`.
    - Assumes `DEFAULT_STARTUP_TIME` and `DEFAULT_WARMUP_TIME` are defined elsewhere in the code.

    Example:
    --------
    df_movements = normalTaxiFuel_df(df_movements, startupTime=300, warmupTime=120)
    """

    df_movements = fuelMESengine_df(
        df_movements, startupTime, warmupTime, colName=colNames[0]
    )
    df_movements = fuelMESapu_df(
        df_movements, startupTime, warmupTime, colName=colNames[1]
    )
    df_movements = fuelTaxiEngine_df(df_movements, colName=colNames[2])

    return df_movements


def extAGPSTaxiFuel_df(df_movements, colNames=["extAGPSapu", "extAGPStug"]):
    """
    Calculates and adds columns for fuel consumption associated with external ground power sources (AGPS)
    during taxiing in the aircraft movements DataFrame.

    This function computes fuel consumption by the Auxiliary Power Unit (APU) when using external ground
    power during taxiing, based on the normal APU fuel flow rate and the taxi duration. Additionally,
    it creates a placeholder column for tug fuel consumption.

    Parameters:
    -----------
    df_movements : pd.DataFrame
        DataFrame containing aircraft movement data, including columns such as 'APUnormalFF' for APU
        fuel flow rate and 'taxiDuration' for the duration of the taxi phase.

    colNames : list of str, optional
        List of column names for storing calculated fuel consumption values related to AGPS.
        Defaults to `['extAGPSapu', 'extAGPStug']`:
        - `colNames[0]`: Fuel consumed by the APU with external AGPS during taxi.
        - `colNames[1]`: Placeholder for tug fuel consumption, currently set to NaN.

    Returns:
    --------
    pd.DataFrame
        Updated DataFrame with two added columns:
        - `extAGPSapu`: Fuel consumption by APU during taxi using external AGPS.
        - `extAGPStug`: Placeholder column for fuel consumed by tugs, currently set as NaN.

    Calculation:
    ------------
    - `extAGPSapu`: Calculated as `APUnormalFF` (APU fuel flow rate) per second, multiplied by the
      total seconds of `taxiDuration`.
    - `extAGPStug`: Set as NaN, providing a placeholder for potential future calculations.

    Example:
    --------
    df_movements = extAGPSTaxiFuel_df(df_movements, colNames=['extAGPSapu', 'extAGPStug'])
    """

    df_movements[colNames[0]] = (
        df_movements["APUnormalFF"]
        / 3600
        * df_movements["taxiDuration"].dt.total_seconds()
    )
    df_movements[colNames[1]] = np.nan

    return df_movements


def getIdleFF(typecode: str) -> float:
    """
    Retrieves the idle fuel flow (FF) rate based on the aircraft type according to the ICAO Aircraft Emission Database.
    Idle fuel flow refers to the fuel consumption rate at approximately 7% thrust, which is typically used for ground idle conditions.

    The function first attempts to retrieve the default engine type and its idle fuel flow rate from the openap.prop.available_aircraft()
    database for the specified aircraft typecode. If the typecode is not available in the database, the function uses predefined engine
    mappings from the AIRCRAFT_INFO dictionary.

    Args:
        typecode (str): The ICAO typecode of the aircraft for which the idle fuel flow rate is to be retrieved.

    Returns:
        float: The idle fuel flow (FF) rate for the specified aircraft type in kilograms per hour (kg/h).

    Raises:
        ValueError: If the aircraft type is not recognized or available in the database or AIRCRAFT_INFO mapping,
                    or if the engine type is not found in the AIRCRAFT_INFO dictionary.

    Notes:
        - The function assumes that the input typecode is a valid ICAO aircraft typecode.
        - If the typecode is not found in the database, the function uses the AIRCRAFT_INFO dictionary,
          which should contain predefined engine mappings for aircraft types not covered in the ICAO Aircraft Emission Database.
        - If the engine type is not found, a ValueError is raised.
        - This function requires the `prop` module from openap to be properly imported and initialized.

    Example:
        >>> idle_ff = getIdleFF('B737')
        >>> print(idle_ff)
        0.8  # Example output, actual value depends on data and typecode
    """

    typecode = typecode.upper()

    try:
        # Check if the aircraft is available in the database
        aircraft = prop.aircraft(typecode)
        engine_type = aircraft["engine"]["default"]
    except:
        # Use predefined engine mapping if the aircraft is not available in the database
        if typecode in AIRCRAFT_INFO:
            engine_type = AIRCRAFT_INFO.get(typecode, {}).get("engine", "Unknown")

    # Retrieve engine properties
    engine = prop.engine(engine_type)

    # Extract idle fuel flow
    ff_idle = engine["ff_idl"]

    return ff_idle


def getIdleFF_df(df_movements):
    """
    Adds engine idle fuel flow (FF) rates to the aircraft movements DataFrame based on the aircraft's type code.

    This function maps each unique `typecode` in the `df_movements` DataFrame to its corresponding
    idle fuel flow rate using the `getIdleFF` function. The mapped idle fuel flow rates are then
    added as a new column, `engIdleFF`, in `df_movements`.

    Parameters:
    -----------
    df_movements : pd.DataFrame
        DataFrame containing aircraft movement data with a `typecode` column that specifies the
        aircraft type code for each flight.

    Returns:
    --------
    pd.DataFrame
        Updated DataFrame with an additional column:
        - `engIdleFF`: Engine idle fuel flow rate associated with each `typecode`.

    Notes:
    ------
    - Assumes that the `getIdleFF` function is defined elsewhere and returns the idle fuel flow rate
      for a given `typecode`.
    - This function uses a dictionary mapping for efficient retrieval of idle fuel flow rates,
      especially useful when `typecode` entries are repeated in `df_movements`.

    Example:
    --------
    df_movements = getIdleFF_df(df_movements)
    """

    unique_typecodes = df_movements.typecode.unique()
    typecode_to_idleFF = {
        typecode: getIdleFF(typecode) for typecode in unique_typecodes
    }
    df_movements["engIdleFF"] = df_movements["typecode"].map(typecode_to_idleFF)

    return df_movements


def getAPUfuelFlow(typecode: str, df_apu=DF_APU, column="") -> pd.Series:
    """
    Retrieves the APU (Auxiliary Power Unit) fuel consumption rates in kilograms per hour for a specified aircraft type.

    The function determines the APU fuel consumption for different operating modes (startup, normal, and high load)
    based on the aircraft's maximum passenger capacity and whether it is classified as an old aircraft. The APU
    fuel consumption data is retrieved from a provided DataFrame.

    Args:
        typecode (str): The ICAO typecode of the aircraft for which APU fuel consumption data is requested.
        df_apu (pd.DataFrame): A DataFrame containing APU fuel consumption data for various aircraft types.
                               Defaults to DF_APU.

    Returns:
        pd.Series: A pandas Series containing the fuel consumption rates for startup, normal, and high load
                   conditions for the specified aircraft type.

    Notes:
        - The function uses predefined classifications to determine whether an aircraft is considered 'old'. This
        information is defined in agps_config.OLD_AIRCRAFT.
        - If the aircraft type is not found in the ICAO database, a lookup is performed in the AIRCRAFT_INFO dictionary.
        - The APU fuel consumption is determined based on the aircraft's maximum passenger capacity and its
          classification as old or new.

    Example:
        >>> apu_fuel = getAPUfuel('A320')
        >>> print(apu_fuel)
        startup    68.0
        normal     101.0
        high       110.0
        Name: A320, dtype: float64
    """

    typecode = typecode.upper()

    # Check if aircraft is considered old
    is_old = typecode in OLD_AIRCRAFT

    # Get maximum passenger count or default to a lookup if not found in prop
    try:
        maxPax = prop.aircraft(typecode)["pax"]["max"]
    except:
        maxPax = AIRCRAFT_INFO.get(typecode, {}).get("max_pax", "Unknown")
        # if maxPax is None:
        #     raise ValueError(f"Aircraft type '{typecode}' not recognized or not available in the database.")

    # Determine the index to use for APU fuel data lookup
    if maxPax < 100:
        idx = 0
    elif 100 <= maxPax < 200:
        idx = 2 if is_old else 1
    elif 200 <= maxPax < 300:
        idx = 3
    else:
        idx = 4 if is_old else 5

    if column == "high":
        apu_FF = df_apu.iloc[idx]["high"]
    elif column == "normal":
        apu_FF = df_apu.iloc[idx]["normal"]
    elif column == "startup":
        apu_FF = df_apu.iloc[idx]["startup"]
    else:
        apu_FF = df_apu.iloc[idx]

    # Return the selected row as a Series from the DataFrame
    return apu_FF


def getAPUfuelFlow_df(df_movements: pd.DataFrame) -> pd.DataFrame:
    """
    Adds Auxiliary Power Unit (APU) fuel flow rates to the aircraft movements DataFrame based on the aircraft type code.

    This function retrieves high and normal APU fuel flow rates for each unique `typecode` in `df_movements`
    using the `getAPUfuelFlow` function and a reference DataFrame `DF_APU`. The results are added as new columns,
    `APUhighFF` and `APUnormalFF`, representing high and normal APU fuel flow rates, respectively.

    Parameters:
    -----------
    df_movements : pd.DataFrame
        DataFrame containing aircraft movement data with a `typecode` column, specifying the aircraft type
        code for each flight.

    Returns:
    --------
    pd.DataFrame
        Updated DataFrame with two additional columns:
        - `APUhighFF`: High APU fuel flow rate for each aircraft `typecode`.
        - `APUnormalFF`: Normal APU fuel flow rate for each aircraft `typecode`.

    Notes:
    ------
    - Assumes that `getAPUfuelFlow` is a function defined elsewhere that takes a `typecode`, a reference DataFrame
      (`DF_APU`), and a mode ('high' or 'normal') to return the corresponding APU fuel flow rate.
    - Efficiently retrieves APU fuel flow rates for each unique `typecode` by storing values in dictionaries, which
      are then mapped to `df_movements`.

    Example:
    --------
    df_movements = getAPUfuelFlow_df(df_movements)
    """

    unique_typecodes = df_movements.typecode.unique()
    typecode_to_APUhigh = {
        typecode: getAPUfuelFlow(typecode, DF_APU, "high")
        for typecode in unique_typecodes
    }
    typecode_to_APUnormal = {
        typecode: getAPUfuelFlow(typecode, DF_APU, "normal")
        for typecode in unique_typecodes
    }
    df_movements["APUhighFF"] = df_movements["typecode"].map(typecode_to_APUhigh)
    df_movements["APUnormalFF"] = df_movements["typecode"].map(typecode_to_APUnormal)

    return df_movements


def getNengine(typecode: str) -> int:
    """
    Retrieves the number of engines for a specified aircraft type.

    This function attempts to find the number of engines for a given aircraft typecode by first checking the
    openap.prop.aircraft(). If the aircraft type is not available in the database, the function falls back to using
    predefined data in the `AIRCRAFT_INFO` dictionary specified in agps_config.py.

    Args:
        typecode (str): The ICAO typecode of the aircraft.

    Returns:
        int: The number of engines for the specified aircraft type.

    Raises:
        ValueError: If the aircraft type is not recognized or not available in the database or the `AIRCRAFT_INFO` mapping.

    Notes:
        - The function converts the input `typecode` to uppercase to ensure consistency in lookups.
        - If the typecode is not found in openap.prop.aircraft(), the function uses the `AIRCRAFT_INFO` dictionary to determine
          the number of engines.
        - If the number of engines is not found in either the database or `AIRCRAFT_INFO`, a ValueError is raised.

    Example:
        >>> n_engines = getNengine('B737')
        >>> print(n_engines)
        2
    """

    # Convert typecode to uppercase to ensure consistent lookup
    typecode = typecode.upper()

    # Try to get the number of engines from the prop module
    try:
        nEngines = prop.aircraft(typecode)["engine"]["number"]
    except:
        # If not in prop.aircraft, try AIRCRAFT_INFO
        nEngines = AIRCRAFT_INFO.get(typecode, {}).get("n_engines", "Unknown")

    return nEngines


def getNengine_df(df_movements: pd.DataFrame) -> pd.DataFrame:
    """
    Adds the number of engines for each aircraft type to the aircraft movements DataFrame based on the type code.

    This function retrieves the number of engines (`nEngines`) for each unique `typecode` in `df_movements`
    using the `getNengine` function. The resulting values are added as a new column, `nEngines`, to
    `df_movements`.

    Parameters:
    -----------
    df_movements : pd.DataFrame
        DataFrame containing aircraft movement data, including a `typecode` column that specifies the
        aircraft type code for each flight.

    Returns:
    --------
    pd.DataFrame
        Updated DataFrame with an additional column:
        - `nEngines`: The number of engines associated with each `typecode`.

    Notes:
    ------
    - Assumes that the `getNengine` function is defined elsewhere and returns the number of engines for a given `typecode`.
    - Uses a dictionary mapping to store engine counts for each unique `typecode`, which is then efficiently
      mapped to `df_movements`.

    Example:
    --------
    df_movements = getNengine_df(df_movements)
    """

    unique_typecodes = df_movements.typecode.unique()
    typecode_to_nEngine = {
        typecode: getNengine(typecode) for typecode in unique_typecodes
    }
    df_movements["nEngines"] = df_movements["typecode"].map(typecode_to_nEngine)

    return df_movements


def getMESduration(
    typecode: str, startupTime=DEFAULT_STARTUP_TIME, warmupTime=DEFAULT_WARMUP_TIME
) -> datetime.timedelta:
    """
    Calculates the total duration of the main engine start (MES) and warm-up period for a specified aircraft type.

    This function computes the duration required to start all engines and perform the necessary warm-up for a
    given aircraft type based on the number of engines and predefined times for startup per engine and warm-up.

    Args:
        typecode (str): The ICAO typecode of the aircraft.
        startupTime (int, optional): The time in seconds required to start one engine. Defaults to DEFAULT_STARTUP_TIME.
        warmupTime (int, optional): The warm-up time in seconds after all engines have started. Defaults to DEFAULT_WARMUP_TIME.

    Returns:
        datetime.timedelta: The total duration of the main engine start and warm-up period in seconds.

    Notes:
        - The function first determines the number of engines for the specified aircraft type using the `getNengine` function.
        - The total duration is calculated as the sum of the time to start all engines (each engine taking `startupTime` seconds)
          and the additional `warmupTime` after all engines are started.
        - The `typecode` is converted to uppercase to ensure consistency in lookups.

    Example:
        >>> mes_duration = getMESduration('A320')
        >>> print(mes_duration)
        0:04:30  # Example output, representing 4 minutes and 30 seconds
    """

    nEngines = getNengine(typecode.upper())

    return datetime.timedelta(seconds=(nEngines * startupTime) + warmupTime)


def fuelMESengine_df(
    df_movements: pd.DataFrame,
    startupTime=DEFAULT_STARTUP_TIME,
    warmupTime=DEFAULT_WARMUP_TIME,
    colName="MESengine",
) -> pd.DataFrame:
    """
    Calculates fuel consumption during main engine startup (MES) for each flight based on idle fuel flow
    and the number of engines, and adds it to the aircraft movements DataFrame.

    This function computes the fuel consumption for each engine during the startup and warmup phases
    based on the number of engines (`nEngines`) and the idle fuel flow rate (`engIdleFF`). It sums the
    fuel consumption for all engines and stores the result in a new column specified by `colName`.

    Parameters:
    -----------
    df_movements : pd.DataFrame
        DataFrame containing aircraft movement data, including columns:
        - `engIdleFF`: Idle fuel flow rate per engine.
        - `nEngines`: Number of engines for each aircraft.

    startupTime : float, optional
        Time in seconds allocated for engine startup (default is `DEFAULT_STARTUP_TIME`).

    warmupTime : float, optional
        Time in seconds allocated for engine warmup (default is `DEFAULT_WARMUP_TIME`).

    colName : str, optional
        Name of the column to store the calculated MES fuel consumption (default is 'MESengine').

    Returns:
    --------
    pd.DataFrame
        Updated DataFrame with an additional column:
        - `colName`: Total MES fuel consumption for each aircraft, considering all engines.

    Calculation:
    ------------
    - Creates an array of engine numbers from 1 up to the maximum number of engines.
    - Calculates fuel consumption for each engine as `engIdleFF * (engine_number * startupTime + warmupTime)`.
    - Uses a mask to exclude fuel consumption for engines beyond the actual `nEngines` count for each row.
    - Sums fuel consumption across all active engines for each flight.

    Example:
    --------
    df_movements = fuelMESengine_df(df_movements, startupTime=300, warmupTime=120, colName='MESengineFuel')
    """

    # Create an array representing the engine numbers from 1 up to the maximum nEngine value
    engine_numbers = np.arange(1, df_movements["nEngines"].max() + 1)

    # Reshape 'engIdleFF' and 'nEngines' to match the broadcasting requirements
    idle_ff_values = df_movements["engIdleFF"].values[:, np.newaxis]
    n_engine_values = df_movements["nEngines"].values[:, np.newaxis]

    # Calculate the fuel consumption for each engine and sum across all engines up to 'nEngines' for each row
    fuel_per_engine = idle_ff_values * (engine_numbers * startupTime + warmupTime)

    # Mask out engines that exceed 'nEngines' for each row
    fuel_per_engine = np.where(engine_numbers <= n_engine_values, fuel_per_engine, 0)

    # Sum the fuel consumption for all engines for each row
    df_movements[colName] = fuel_per_engine.sum(axis=1)

    return df_movements


def fuelMESapu_df(
    df_movements: pd.DataFrame,
    startupTime=DEFAULT_STARTUP_TIME,
    warmupTime=DEFAULT_WARMUP_TIME,
    colName="MESapu",
) -> pd.DataFrame:
    """
    Calculates fuel consumption by the Auxiliary Power Unit (APU) during the main engine startup (MES) phase
    for each flight and adds it to the aircraft movements DataFrame.

    This function computes the APU fuel consumption during both the startup and warmup phases using the high
    and normal APU fuel flow rates (`APUhighFF` and `APUnormalFF`). The result is added as a new column
    specified by `colName`.

    Parameters:
    -----------
    df_movements : pd.DataFrame
        DataFrame containing aircraft movement data, including columns:
        - `nEngines`: Number of engines for each aircraft.
        - `APUhighFF`: High fuel flow rate for the APU during engine startup.
        - `APUnormalFF`: Normal fuel flow rate for the APU during the warmup phase.

    startupTime : float, optional
        Time in seconds allocated for the APU's high fuel flow phase during engine startup
        (default is `DEFAULT_STARTUP_TIME`).

    warmupTime : float, optional
        Time in seconds allocated for the APU's normal fuel flow phase during engine warmup
        (default is `DEFAULT_WARMUP_TIME`).

    colName : str, optional
        Name of the column to store the calculated APU fuel consumption during the MES phase
        (default is 'MESapu').

    Returns:
    --------
    pd.DataFrame
        Updated DataFrame with an additional column:
        - `colName`: Total APU fuel consumption during MES for each flight.

    Calculation:
    ------------
    - Computes high APU fuel consumption as `(nEngines * APUhighFF / 3600) * startupTime`, representing
      fuel used during the high-power startup phase.
    - Computes normal APU fuel consumption as `(APUnormalFF / 3600) * warmupTime`, representing fuel used
      during the warmup phase.
    - Sums the two components to obtain total APU fuel consumption during the MES phase.

    Example:
    --------
    df_movements = fuelMESapu_df(df_movements, startupTime=300, warmupTime=120, colName='MESapuFuel')
    """

    df_movements[colName] = (
        df_movements["nEngines"] * df_movements["APUhighFF"] / 3600 * startupTime
    ) + (df_movements["APUnormalFF"] / 3600 * warmupTime)

    return df_movements


def fuelTaxiEngine_df(
    df_movements: pd.DataFrame, singleEngineTaxi=False, colName="normTAXIengine"
) -> pd.DataFrame:
    """
    Calculates fuel consumption during the taxiing phase for each flight and adds it to the aircraft movements DataFrame.

    This function computes the fuel used by the engines during taxiing based on the taxi duration,
    number of engines (`nEngines`), and idle fuel flow rate (`engIdleFF`). If `singleEngineTaxi` is set
    to True, the function assumes that only one engine is used during taxiing, reducing the total fuel
    consumption by half.

    Parameters:
    -----------
    df_movements : pd.DataFrame
        DataFrame containing aircraft movement data, including columns:
        - `taxiDuration`: Duration of the taxiing phase.
        - `nEngines`: Number of engines for each aircraft.
        - `engIdleFF`: Idle fuel flow rate per engine.

    singleEngineTaxi : bool, optional
        If True, calculates taxi fuel consumption based on single-engine taxiing by reducing
        the fuel consumption by half (default is False).

    colName : str, optional
        Name of the column to store the calculated taxi fuel consumption (default is 'normTAXIengine').

    Returns:
    --------
    pd.DataFrame
        Updated DataFrame with an additional column:
        - `colName`: Total taxi fuel consumption for each flight, considering either single or dual-engine taxiing.

    Calculation:
    ------------
    - Calculates fuel consumption as `taxiDuration * nEngines * engIdleFF`, where:
      - `taxiDuration` is the total taxi time in seconds.
      - `nEngines` is the number of engines.
      - `engIdleFF` is the idle fuel flow rate per engine.
    - If `singleEngineTaxi` is True, the calculated fuel consumption is halved.

    Example:
    --------
    df_movements = fuelTaxiEngine_df(df_movements, singleEngineTaxi=True, colName='taxiFuelConsumption')
    """

    fuelConsumption = (
        df_movements["taxiDuration"].dt.total_seconds()
        * df_movements["nEngines"]
        * df_movements["engIdleFF"]
    )

    if singleEngineTaxi:
        fuelConsumption = fuelConsumption * 0.5

    df_movements[colName] = fuelConsumption

    return df_movements


def convert_timestamp(x):
    """
    Converts various input formats to a standardized pandas Timestamp.

    This function ensures consistent timestamp formatting across different data sources by converting
    supported input types (Unix timestamps, pandas Timestamps) to pandas `Timestamp` objects. Invalid
    or unsupported inputs are converted to `pd.NaT`.

    Parameters:
    -----------
    x : Any
        The input value to be converted to a pandas `Timestamp`. Supported formats include:
        - Unix timestamps (float or int) that represent seconds since the epoch (e.g., 1.6e9).
        - pandas `Timestamp` objects, which are returned unchanged.

    Returns:
    --------
    pd.Timestamp or pd.NaT
        A pandas `Timestamp` if the conversion is successful, otherwise `pd.NaT`.

    Conversion Logic:
    -----------------
    - Unix timestamps (float or int) greater than 1e9 are converted using `pd.to_datetime()`.
    - `pd.Timestamp` inputs are returned unchanged for compatibility with existing pandas structures.
    - All other inputs or invalid conversions result in `pd.NaT`.

    Examples:
    ---------
    >>> convert_timestamp(1.6e9)
    Timestamp('2020-09-13 12:26:40')

    >>> convert_timestamp(pd.Timestamp('2025-03-28 10:00:00'))
    Timestamp('2025-03-28 10:00:00')

    >>> convert_timestamp("invalid_input")
    NaT
    """
    if (
        isinstance(x, (float, int)) and x > 1e9
    ):  # Convert to unix timestamps if float or int.
        return pd.to_datetime(x, unit="s", errors="coerce")
    elif isinstance(
        x, pd.Timestamp
    ):  # If x is alreardy a pandas timestamp, x will be returned
        return x
    else:
        return pd.NaT  # Invalid Value will be converted to Not a Timestamp


def ensure_traffic_datatypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures consistent and appropriate data types for columns in a trajectory DataFrame.

    This function standardizes the data types of a DataFrame derived from a `traffic` object.
    It ensures compatibility with downstream processing by converting columns to expected data types
    (e.g., numeric, boolean, timestamp, and object/string).

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing trajectory data obtained from a `traffic` object.
        The DataFrame is expected to have specific columns, including timestamps, numeric attributes,
        and categorical strings.

    Returns:
    --------
    pd.DataFrame
        The DataFrame with columns converted to their expected data types.

    Expected Columns and Their Data Types:
    --------------------------------------
    - 'timestamp' (Timestamp): Converted using `convert_timestamp()` for consistency.
    - 'hour' (Timestamp): Converted using `convert_timestamp()` for consistency.
    - 'onground' (bool): Converted to boolean type.
    - 'alert' (bool): Converted to boolean type.
    - 'spi' (bool): Converted to boolean type.
    - 'last_position' (float): Converted to 64-bit floating-point format.
    - 'latitude' (float): Converted to double-precision floating-point format.
    - 'longitude' (float): Converted to double-precision floating-point format.
    - 'groundspeed' (float): Converted to double-precision floating-point format.
    - 'track' (float): Converted to double-precision floating-point format.
    - 'vertical_rate' (float): Converted to double-precision floating-point format.
    - 'altitude' (float): Converted to double-precision floating-point format.
    - 'geoaltitude' (float): Converted to double-precision floating-point format.
    - 'lastcontact' (float): Converted to double-precision floating-point format.
    - 'icao24' (str): Converted to string/object type.
    - 'callsign' (str): Converted to string/object type.
    - 'squawk' (str): Converted to string/object type.

    Notes:
    ------
    - It is assumed that `convert_timestamp()` is a custom function that converts various timestamp formats to a standard pandas `Timestamp`.
    - This function helps ensure compatibility and consistency when working with trajectory data processed by the `traffic` library.

    Example:
    --------
    >>> df_cleaned = ensure_traffic_datatypes(df)
    """

    # # Make sure data types of downloaded data are correct
    df["timestamp"] = df["timestamp"].apply(convert_timestamp)
    df["hour"] = df["hour"].apply(convert_timestamp)
    df["onground"] = df["onground"].astype(np.bool_)
    df["alert"] = df["alert"].astype(np.bool_)
    df["spi"] = df["spi"].astype(np.bool_)
    df["last_position"] = df["last_position"].astype(np.float64)

    df["latitude"] = df["latitude"].astype("double")
    df["longitude"] = df["longitude"].astype("double")
    df["groundspeed"] = df["groundspeed"].astype("double")
    df["track"] = df["track"].astype("double")
    df["vertical_rate"] = df["vertical_rate"].astype("double")
    df["altitude"] = df["altitude"].astype("double")
    df["geoaltitude"] = df["geoaltitude"].astype("double")
    df["lastcontact"] = df["lastcontact"].astype("double")

    df["icao24"] = df["icao24"].astype("object")
    df["callsign"] = df["callsign"].astype("object")
    df["squawk"] = df["squawk"].astype("object")

    return df
