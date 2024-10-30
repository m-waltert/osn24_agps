from agps_config import AIRCRAFT_INFO, DEFAULT_STARTUP_TIME, DEFAULT_WARMUP_TIME, DF_APU, OLD_AIRCRAFT, AC2CONSIDER, DF_MISSING_ICAO24
import traffic
from traffic.core import Traffic
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
data_proj = ccrs.PlateCarree()


def get_lon_lat_by_runway(df_runways, runway_name):
    """
    Retrieves the longitude and latitude coordinates for a specified runway.

    This function searches a DataFrame containing runway information for a row 
    matching the provided `runway_name`. If found, it extracts and returns the 
    longitude and latitude of that runway; otherwise, it returns `None` values.
    """

    row = df_runways[df_runways['name'] == runway_name]
    if not row.empty:
        lat = row.iloc[0]['latitude']
        lon = row.iloc[0]['longitude']
        return lon, lat
    else:
        return None, None
    

# Function to extend a point in the direction of another point
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
    if runway[-1] == 'L':
        suffix = 'R'
    elif runway[-1] == 'R':
        suffix = 'L'
    elif runway[-1] == 'C':
        suffix = 'C'
    else:
        suffix = ''

    # Combine the opposite number with the suffix
    return f"{opposite_number_str}{suffix}"


def get_Box_around_Rwy(Rwy: str,
                       airport_str: str,
                       extension_distance=0.002,
                       extension_width = 0.0004):
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
    rwy_box = LineString([extend_point(point2, point1, extension_distance),
                          extend_point(point1, point2, extension_distance)])

    # Extend width by buffering
    rwy_box = rwy_box.buffer(distance=extension_width, cap_style='square')
    return rwy_box


def takeoff_detection(traj,
                       airport_str = 'LSZH'
                       ) -> str:
    """
    Detects takeoff events from a given aircraft trajectory and updates trajectory data.

    This function determines if a given trajectory corresponds to a takeoff at a specified airport. 
    If a takeoff is detected, the runway used for takeoff and the lineup time are identified and 
    appended to the trajectory data.

    Args:
        traj (Trajectory): The trajectory object containing flight data, including positions 
                           and timestamps.
        airport_str (str, optional): The ICAO code of the airport to detect takeoff from.
                                     Defaults to 'LSZH' (Zurich Airport).

    Returns:
        Trajectory: The input trajectory with updated attributes:
            - 'lineupTime2' (float): The time of lineup on the runway, or NaN if no takeoff.
            - 'takeoffRunway2' (str): The runway used for takeoff, or an empty string if none.
            - 'isTakeoff2' (bool): Boolean flag indicating whether a takeoff was detected.
    
    Notes:
        - The function uses a runway geometry box to detect whether the trajectory aligns with 
          the takeoff process.
        - The trajectory is clipped to the box around the runway for better accuracy in detecting 
          the takeoff.

    Example:
        >>> updated_traj = takeoff_detection(traj, airport_str='LSZH')
    """
    
    lineupTime = np.nan
    isTakeoff = False

    if ~traj.landing_at(airport_str):

        # Classification of takeoff using new routine
        takeoffRunway = traj.get_toff_runway(airport_str)

        if takeoffRunway is not None:
            isTakeoff = True

            # Get runway geometry (box around runway)
            rwy_box = get_Box_around_Rwy(takeoffRunway, airport_str)

            clipped_traj = traj.clip(rwy_box)

            if clipped_traj is not None and not clipped_traj.data.empty:
                lineupTime = clipped_traj.start
        else:
            takeoffRunway=''

    traj.data = traj.data.copy()
    traj.data.loc[:, 'lineupTime'] = lineupTime
    traj.data.loc[:, 'takeoffRunway'] = takeoffRunway
    traj.data.loc[:, 'isTakeoff'] = isTakeoff

    return traj


def alternative_pushback_detection(traj, standAreas, airport_str='LSZH'):
    """
    Detects and calculates pushback and taxiing characteristics for a given aircraft trajectory during takeoff.

    This function determines whether an aircraft has performed a pushback before taxiing by checking if the 
    trajectory intersects defined stand areas. If pushback is detected, it calculates the start and duration of 
    pushback, the start time of taxiing, as well as taxi duration and distance. It also updates the trajectory 
    data with pushback and taxi attributes.

    Parameters:
    -----------
    traj : Trajectory
        The aircraft trajectory object containing time series data of the aircraft's ground movement and status.
    
    standAreas : list of Polygon
        List of polygonal areas representing possible stand locations where aircraft might initiate pushback.
    
    airport_str : str, optional
        String representing the airport code (default is 'LSZH') for location-specific data handling.

    Returns:
    --------
    traj : Trajectory
        Updated trajectory object with additional attributes:
        - isPushback (bool): Whether the aircraft performed a pushback.
        - startPushback (datetime): The start time of the pushback maneuver, if detected.
        - startTaxi (datetime): The start time of the taxiing phase.
        - pushbackDuration (timedelta): Duration of the pushback phase.
        - taxiDuration (timedelta): Duration of the taxiing phase.
        - taxiDistance (float): Total distance covered during taxiing in meters.

    Notes:
    ------
    - The function is specifically designed for analyzing takeoff events.
    - If the ground coverage is incomplete or trajectory segments are missing, 
      adjustments may be made to the pushback and taxi timings.
    - The calculated `taxiDuration` and `taxiDistance` are only valid for takeoff movements 
      with available lineup times.

    Example:
    --------
    traj = alternative_pushback_detection(traj, standAreas, airport_str='LSZH')
    """


    lineupTime = traj.data.lineupTime.iloc[0]
    taxiDuration = np.nan
    taxiDistance = np.nan
    startPushback = np.nan
    pushbackDuration = np.nan
    startTaxi = np.nan
    isPushback = False

    # Apply pushback and taxi detection only to takeoffs
    if traj.data.isTakeoff.iloc[0]:


        # Pushback Detection
        for i, standArea in enumerate(standAreas):
            clipped_traj = traj.clip(standArea)

            # Check whether traj is inside stand_area
            if (clipped_traj is None) or (clipped_traj.data.empty):
                        continue
            else:
                isPushback = True
                break   
        
        if isPushback and (clipped_traj is not None):

            df = traj.data
            nonzeroGS = df.compute_gs.rolling(5).median() > 1
            leaveStandTime = clipped_traj.stop
            leaveStandTimeIndex = df[df['timestamp']==leaveStandTime].index[0]

            # Find the start of the pushback
            startPushbackIndex = leaveStandTimeIndex
            while startPushbackIndex > 0 and nonzeroGS.loc[startPushbackIndex - 1]:
                startPushbackIndex -= 1

            # Find the end of the pushback
            endPushbackIndex = leaveStandTimeIndex
            while endPushbackIndex < df.index.max() and nonzeroGS.loc[endPushbackIndex + 1]:
                endPushbackIndex += 1

            startPushback = df.loc[startPushbackIndex, 'timestamp']
            startTaxi = df.loc[endPushbackIndex, 'timestamp']

            # If start and end of pushback are identical
            if startPushback == startTaxi:
                startPushback = startPushback - timedelta(seconds=2)

            # Pushback duration
            pushbackDuration = startTaxi - startPushback

            # Check if taxiDuration is less than 0 seconds. This can happen if the ground coverage is not perfect and parts of the traj are missing.
            if (lineupTime - startTaxi) < timedelta(seconds=0):
                startTaxi = leaveStandTime
                isPushback = False
                startPushback = np.nan
                pushbackDuration = np.nan

        # If takeoff is not pushback, check for stand
        else:

            parkingPosition = traj.on_parking_position(airport_str).max()

            if (parkingPosition is not None) and (parkingPosition.duration > timedelta(seconds=30)):
                startTaxi = parkingPosition.stop
            # It it is a takeoff, but neither a pushback nor a stand can be detected
            else:
                startTaxi = traj.start if not traj.data.empty else np.nan

        # Calculate taxiDuration
        taxiDuration = lineupTime - startTaxi

        # Calcualte taxiDistance
        if (taxiDuration is not np.nan) and (taxiDuration > timedelta(seconds=0)):
            taxi = traj.between(startTaxi, lineupTime)
            taxiDistance = taxi.data.cumdist.iloc[-1] - taxi.data.cumdist.iloc[0]
        
    # Write data to traj
    traj.data = traj.data.copy()
    traj.data.loc[:, 'isPushback'] = isPushback
    traj.data.loc[:, 'startPushback'] = startPushback
    traj.data.loc[:, 'startTaxi'] = startTaxi
    traj.data.loc[:, 'pushbackDuration'] = pushbackDuration
    traj.data.loc[:, 'taxiDuration'] = taxiDuration
    traj.data.loc[:, 'taxiDistance'] = taxiDistance

    return traj


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
    df_movements = df_movements.merge(DF_MISSING_ICAO24[['icao24', 'typecode', 'icaoaircrafttype']], 
                                    on='icao24', 
                                    how='left')

    df_movements['typecode_x'] = df_movements['typecode_x'].combine_first(df_movements['typecode_y'])
    # df_movements['icaoaircrafttype_x'] = df_movements['icaoaircrafttype_x'].combine_first(df_movements['icaoaircrafttype_y'])
    df_movements.loc[df_movements['icaoaircrafttype_y'].notna(), 'icaoaircrafttype_x'] = df_movements['icaoaircrafttype_y']
    df_movements.drop(columns=['typecode_y', 'icaoaircrafttype_y'], inplace=True)

    # Rename specific columns
    df_movements = df_movements.rename(columns={'typecode_x': 'typecode', 
                                                'icaoaircrafttype_x': 'icaoaircrafttype'})
    
    # Replace 'C68A' with 'L2J' in the 'icaoaircrafttype' column 
    df_movements.loc[df_movements['icaoaircrafttype'] == 'C68A', 'icaoaircrafttype'] = 'L2J'


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

    # Filter out rows where 'icaoaircrafttype' contains 'H' anywhere in the string
    df_movements = df_movements[~df_movements['icaoaircrafttype'].str.contains('H', case=False, na=False)]

    # Filter out trajectories which have taxi distances of NaN
    df_movements = df_movements[~df_movements['taxiDistance'].isna()]

    # Filter out trajectories which have taxi distances of NaN
    df_movements = df_movements[~df_movements['lineupTime'].isna()]

    # Filter trajectories which have taxiDurations of less than 0
    df_movements =df_movements[~(df_movements['taxiDuration'] < pd.Timedelta(0))]

    return df_movements


def get_df_movements(gnd_trajs) -> pd.DataFrame:
    """
    Summarizes ground trajectory data by grouping information by flight ID and extracting key 
    attributes for each unique flight to create a clean, summarized DataFrame of movements.

    This function filters out flights without takeoff runway information, groups the data by 
    `flight_id`, and retrieves essential fields for each flight, such as `icao24`, `callsign`, 
    pushback and taxi timings, taxi duration and distance, and takeoff runway.

    Parameters:
    -----------
    gnd_trajs : pd.DataFrame
        DataFrame containing ground trajectory data with columns including 'flight_id', 'icao24', 
        'callsign', 'isTakeoff', 'isPushback', 'startPushback', 'startTaxi', 'lineupTime', 
        'taxiDuration', 'taxiDistance', 'takeoffRunway', 'typecode', and 'icaoaircrafttype'.

    Returns:
    --------
    pd.DataFrame
        Summarized DataFrame `df_movements` containing one row per flight with the following columns:
        - 'flight_id': Unique identifier for each flight.
        - 'icao24': Aircraft identifier.
        - 'callsign': Flight callsign.
        - 'isTakeoff': Boolean indicating whether the flight is a takeoff.
        - 'isPushback': Boolean indicating whether pushback was performed.
        - 'startPushback': Timestamp of the start of pushback.
        - 'startTaxi': Timestamp of the start of taxiing.
        - 'lineupTime': Timestamp of the lineup for takeoff.
        - 'taxiDuration': Duration of the taxiing phase.
        - 'taxiDistance': Distance covered during taxiing.
        - 'takeoffRunway': The runway used for takeoff.
        - 'typecode': Type code of the aircraft.
        - 'icaoaircrafttype': ICAO aircraft type code.

    Notes:
    ------
    - Flights without a specified `takeoffRunway` are filtered out.
    - Uses the first available value in each grouped column for each flight.
    - Ensures that the resulting DataFrame has a clean, reset index.

    Example:
    --------
    df_movements = get_df_movements(gnd_trajs)
    """

    # Group by 'flight_id'
    grouped = gnd_trajs.query('takeoffRunway != ""').groupby('flight_id')
    
    # Create a new DataFrame df_movements to store the summarized data
    df_movements = pd.DataFrame()
    
    # Extract the required information
    df_movements['flight_id'] = grouped['flight_id'].first()
    df_movements['icao24'] = grouped['icao24'].first()
    df_movements['callsign'] = grouped['callsign'].first()
    df_movements['isTakeoff'] = grouped['isTakeoff'].first()
    df_movements['isPushback'] = grouped['isPushback'].first()
    df_movements['startPushback'] = grouped['startPushback'].first()
    df_movements['startTaxi'] = grouped['startTaxi'].first()
    df_movements['lineupTime'] = grouped['lineupTime'].first()
    df_movements['taxiDuration'] = grouped['taxiDuration'].first()
    df_movements['taxiDistance'] = grouped['taxiDistance'].first()
    df_movements['takeoffRunway'] = grouped['takeoffRunway'].first()
    df_movements['typecode'] = grouped['typecode'].first()
    df_movements['icaoaircrafttype'] = grouped['icaoaircrafttype'].first()
    
    # Reset index to get a clean DataFrame
    df_movements = df_movements.reset_index(drop=True)

    return df_movements


def normalTaxiFuel_df(df_movements: pd.DataFrame, 
                      startupTime=DEFAULT_STARTUP_TIME, 
                      warmupTime=DEFAULT_WARMUP_TIME, 
                      colNames=['MESengine', 'MESapu', 'normTAXIengine']
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

    df_movements = fuelMESengine_df(df_movements, startupTime, warmupTime, colName=colNames[0])
    df_movements = fuelMESapu_df(df_movements, startupTime, warmupTime, colName=colNames[1])
    df_movements = fuelTaxiEngine_df(df_movements, colName=colNames[2])

    return df_movements


def extAGPSTaxiFuel_df(df_movements, colNames=['extAGPSapu', 'extAGPStug']):
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

    df_movements[colNames[0]] = df_movements['APUnormalFF'] / 3600 * df_movements['taxiDuration'].dt.total_seconds()
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
        engine_type = aircraft['engine']['default']
    except:
        # Use predefined engine mapping if the aircraft is not available in the database
        if typecode in AIRCRAFT_INFO:
            engine_type = AIRCRAFT_INFO.get(typecode, {}).get('engine', 'Unknown')

    # Retrieve engine properties
    engine = prop.engine(engine_type)
    
    # Extract idle fuel flow
    ff_idle = engine['ff_idl']


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
    typecode_to_idleFF = {typecode: getIdleFF(typecode) for typecode in unique_typecodes}
    df_movements['engIdleFF'] = df_movements['typecode'].map(typecode_to_idleFF)

    return df_movements


def getAPUfuelFlow(typecode: str, df_apu = DF_APU, column='') -> pd.Series:
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
        maxPax = prop.aircraft(typecode)['pax']['max']
    except:
        maxPax = AIRCRAFT_INFO.get(typecode, {}).get('max_pax', 'Unknown')
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


    if column == 'high':
        apu_FF = df_apu.iloc[idx]['high']
    elif column == 'normal':
        apu_FF = df_apu.iloc[idx]['normal']
    elif column == 'startup':
        apu_FF = df_apu.iloc[idx]['startup']
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
    typecode_to_APUhigh = {typecode: getAPUfuelFlow(typecode, DF_APU, 'high') for typecode in unique_typecodes}
    typecode_to_APUnormal = {typecode: getAPUfuelFlow(typecode, DF_APU, 'normal') for typecode in unique_typecodes}
    df_movements['APUhighFF'] = df_movements['typecode'].map(typecode_to_APUhigh)
    df_movements['APUnormalFF'] = df_movements['typecode'].map(typecode_to_APUnormal)

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
        nEngines = prop.aircraft(typecode)['engine']['number']
    except:
        # If not in prop.aircraft, try AIRCRAFT_INFO
        nEngines = AIRCRAFT_INFO.get(typecode, {}).get('n_engines', 'Unknown')

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
    typecode_to_nEngine = {typecode: getNengine(typecode) for typecode in unique_typecodes}
    df_movements['nEngines'] = df_movements['typecode'].map(typecode_to_nEngine)

    return df_movements


def getMESduration(typecode: str, 
                   startupTime=DEFAULT_STARTUP_TIME, 
                   warmupTime=DEFAULT_WARMUP_TIME) -> datetime.timedelta:
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

    return datetime.timedelta(seconds = (nEngines * startupTime) + warmupTime)


def fuelMESengine_df(df_movements: pd.DataFrame, 
                     startupTime=DEFAULT_STARTUP_TIME, 
                     warmupTime=DEFAULT_WARMUP_TIME,
                     colName='MESengine') -> pd.DataFrame:
    
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
    engine_numbers = np.arange(1, df_movements['nEngines'].max() + 1)

    # Reshape 'engIdleFF' and 'nEngines' to match the broadcasting requirements
    idle_ff_values = df_movements['engIdleFF'].values[:, np.newaxis]
    n_engine_values = df_movements['nEngines'].values[:, np.newaxis]

    # Calculate the fuel consumption for each engine and sum across all engines up to 'nEngines' for each row
    fuel_per_engine = idle_ff_values * (engine_numbers * startupTime + warmupTime)

    # Mask out engines that exceed 'nEngines' for each row
    fuel_per_engine = np.where(engine_numbers <= n_engine_values, fuel_per_engine, 0)

    # Sum the fuel consumption for all engines for each row
    df_movements[colName] = fuel_per_engine.sum(axis=1)

    return df_movements


def fuelMESapu_df(df_movements: pd.DataFrame,
                  startupTime=DEFAULT_STARTUP_TIME,
                  warmupTime=DEFAULT_WARMUP_TIME,
                  colName='MESapu') -> pd.DataFrame:
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

    df_movements[colName] = (df_movements['nEngines'] * df_movements['APUhighFF'] / 3600 * startupTime) + (df_movements['APUnormalFF'] / 3600 * warmupTime)

    return df_movements


def fuelTaxiEngine_df(df_movements: pd.DataFrame, 
                      singleEngineTaxi=False,
                      colName='normTAXIengine') -> pd.DataFrame:
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

    fuelConsumption = (df_movements['taxiDuration'].dt.total_seconds() 
                       * df_movements['nEngines'] 
                       * df_movements['engIdleFF'])

    if singleEngineTaxi:
        fuelConsumption = fuelConsumption * 0.5

    df_movements[colName] = fuelConsumption

    return df_movements 