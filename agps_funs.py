from agps_config import AIRCRAFT_INFO, DEFAULT_STARTUP_TIME, DEFAULT_WARMUP_TIME, DF_APU, OLD_AIRCRAFT, AC2CONSIDER
import traffic
from traffic.core import Traffic
from traffic.data import airports
import numpy as np
from datetime import timedelta
from shapely.geometry import base
import pandas as pd
import datetime
import os
from openap import prop

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from cartes.crs import EuroPP
import cartopy.crs as ccrs
data_proj = ccrs.PlateCarree()

# def my_intersect(flight, shape):
#     """Adaptation of def _flight_intersects() from airspace.py
#     """
#     # if "altitude" in flight.data.columns:
#     #     flight = flight.query("altitude.notnull()")  # type: ignore
#     if flight is None or (linestring := flight.linestring) is None:
#         return False
#     if isinstance(shape, base.BaseGeometry):
#         bla = not linestring.intersection(shape).is_empty
#         return (not linestring.intersection(shape).is_empty)
#     return False


def takeoff_detection(traj, 
                      rwy_geometries, 
                      df_rwys, 
                      airport_str='LSZH',
                      gsColName='compute_gs'):
    """
    Detects the takeoff event from a given trajectory and associates the takeoff with the runway of an airport.

    This function analyzes a flight trajectory to determine if a takeoff has occurred at a specified airport.
    It uses runway geometries and other flight data to detect takeoff events and identify the runway used 
    for takeoff. If a takeoff is detected, the function adds relevant information to the trajectory data, 
    including the lineup time, takeoff runway, and a boolean flag indicating whether a takeoff was detected.

    Args:
        traj (Trajectory): A traffic flight object
        rwy_geometries (list of shapely.geometry.Polygon): A list of runway geometries (polygons) representing 
                                                           the airport's runways.
        df_rwys (pandas.DataFrame): A DataFrame containing runway information for the specified airport, 
                                    including runway names and bearings.
        airport_str (str): The ICAO code of the airport to check for takeoff. Defaults to 'LSZH'.
        gsColName (str, optional): The name of the column in `traj` that contains ground speed data. Defaults to 'compute_gs'.

    Returns:
        Trajectory: The input `Trajectory` object with additional columns:
                    - 'lineupTime': The timestamp of the aircraft's lineup on the runway, or NaN if no takeoff was detected.
                    - 'takeoffRunway': The name of the detected takeoff runway, or an empty string if no takeoff was detected.
                    - 'isTakeoff': A boolean indicating whether a takeoff was detected.

    Notes:
        - The function assumes that the input `Trajectory` contains data relevant to the specified airport.
        - The function checks each runway geometry to see if the trajectory intersects it, suggesting a possible takeoff.
        - For a takeoff to be confirmed:
          1. The trajectory must intersect with a runway geometry for at least 45 seconds.
          2. The distance from the part of the trajectory clipped to the corresponding runway geometry must not exceed 200 meters.
          3. The ground speed in the first 5 seconds of the clipped part must be less than 30 knots.
          4. The vertical rate in the last 5 seconds of the clipped part must be greater than 500 feet per minute, or the change in 
             cumulative distance per second must exceed 0.0277 nautical miles (equivalent to 100 knots).

    Example:
        >>> traj = takeoff_detection(traj, rwy_geometries, df_rwys, airport_str='LSZH', gsColName='compute_gs')
    """
     
    takeoffRunway = ''
    lineupTime = np.nan
    isTakeoff = False

    if traj.takeoff_from(airport_str):

        for i, rwy in enumerate(rwy_geometries):
            # Intersection traj with rwy -> modification of flight.intersect() from traffic library
            intersection = False
            if traj is None or (linestring := traj.linestring) is None:
                intersection = False
            if isinstance(rwy, base.BaseGeometry):
                intersection = (not linestring.intersection(rwy).is_empty)

            if intersection:

                # Clip traj to runway geometry
                clipped_traj = traj.clip(rwy)

                if (clipped_traj is None) or (clipped_traj.data.empty): #or (clipped_traj.duration < timedelta(seconds=60)):
                    continue

                # Clipped trajs must be in rwy geometry for longer than 45 seconds
                if clipped_traj.duration < timedelta(seconds=45):
                    continue

                # Check maximum distance of clipped traj to rwy geometry. If distance it "too" large, this mean that clipped traj moves away from
                # the runway. This can happen, for instance,  with the rwy10/28 geometry with aircraft departing on runway 16 that cross runway 
                # 10/28 before takeoff
                maxDistfromRwy = clipped_traj.distance(rwy).data.distance.max()
                if np.abs(maxDistfromRwy) > 200:
                    continue

                # Cache traj snippets
                first_5sec = clipped_traj.first(seconds=5).data
                last_5sec = clipped_traj.last(seconds=5).data
                last_20sec = clipped_traj.last(seconds=20).data
                #last_60min_data = traj.last(minutes=60)

                # Calculate ground speed and vertical rate
                median_gs = np.nanmedian(first_5sec[gsColName]) if not first_5sec[gsColName].isna().all() else np.nan
                median_rate = np.nanmedian(last_5sec.vertical_rate) if not last_5sec.vertical_rate.isna().all() else np.nan
                median_cumdistDiff = np.nanmedian(last_5sec.cumdist.diff()) if not last_5sec.cumdist.isna().all() else np.nan

                # It is a take-off if:
                # median_gs in first 5 seconds is less than 30kt, AND
                # (median vertical speed > 500ft/min OR change in cumulative distance per second > 0.0277nm, which is equal to 100kt) in the last 5 seconds
                if (median_gs < 30) and ((median_rate > 500) or median_cumdistDiff > 0.0277) and not last_20sec.empty:
                    isTakeoff = True

                    # Mean track during take-off
                    # median_track = last_20sec.track.median()
                    median_track = last_20sec.compute_track.median()

                    # Line-up time
                    lineupTime = clipped_traj.start

                    # Find the takeoff runway
                    runwayBearings = df_rwys.iloc[2*i:2*i+2].bearing
                    idx = (runwayBearings - median_track).abs().idxmin()
                    takeoffRunway = df_rwys.name.loc[idx]

                    break

    traj.data.loc[:, 'lineupTime'] = lineupTime
    traj.data.loc[:, 'takeoffRunway'] = takeoffRunway
    traj.data.loc[:, 'isTakeoff'] = isTakeoff

    return traj


def alternative_pushback_detection(traj, standAreas, airport_str='LSZH'):
    """
    Detects pushback events for a given flight trajectory using a method different from the one specified in the traffic library.
    This detection is applicable only for trajectories classified as takeoffs using the `takeoff_detection()` method.

    The function determines whether a pushback has occurred based on the flight's interaction with user-provided stand area geometries. 
    These geometries are polygons representing parking or stand locations at an airport. The method calculates the pushback duration, 
    taxi duration, and marks the relevant timestamps in the trajectory data.

    Args:
        traj (Trajectory): A `Trajectory` object representing a flight, containing positional and velocity data.
        standAreas (list of shapely.geometry.Polygon): A list of stand area geometries (polygons) representing parking
                                                       or stand locations at the airport.
        airport_str (str): The ICAO code of the airport to check for pushback and taxi events. Defaults to 'LSZH'.

    Returns:
        Trajectory: The input `Trajectory` object with additional columns:
                    - 'isPushback': A boolean indicating whether a pushback was detected.
                    - 'startPushback': The timestamp when the pushback started, or NaN if not detected.
                    - 'startTaxi': The timestamp when taxiing started, or NaN if not detected.
                    - 'pushbackDuration': The duration of the pushback, or NaN if not detected.
                    - 'taxiDuration': The duration of the taxi, or NaN if not detected.

    Notes:
        - This function assumes that the input trajectory corresponds to a takeoff event as classified using the `takeoff_detection()` method.
        - Pushback detection is based on the aircraft's interaction with defined stand areas:
          - The function first checks if the aircraft intersects with any of the provided stand area polygons.
          - If an intersection is found and the trajectory remains in the area for a significant duration, it is flagged as a pushback.
        - The function calculates the pushback start and end times based on ground speed data:
          - The start of pushback is detected when the aircraft starts moving (ground speed > 1 knot).
          - The end of pushback is detected when the aircraft stops moving (ground speed <= 1 knot).
        - The function calculates the taxi duration from the end of the pushback to the lineup time.
        - If no pushback is detected but a stand position is identified, the start of taxiing is set to the end of the parking duration.

    Example:
        >>> traj = alternative_pushback_detection(traj, standAreas, airport_str='LSZH')
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

        # TaxiTime in Movement


        # TaxiTime NoMovement
        
    # Write data to traj
    traj.data.loc[:, 'isPushback'] = isPushback
    traj.data.loc[:, 'startPushback'] = startPushback
    traj.data.loc[:, 'startTaxi'] = startTaxi
    traj.data.loc[:, 'pushbackDuration'] = pushbackDuration
    traj.data.loc[:, 'taxiDuration'] = taxiDuration
    traj.data.loc[:, 'taxiDistance'] = taxiDistance

    return traj


def normalTaxiFuel(traj):

    MESengine = np.nan
    MESapu = np.nan
    normTAXIengine = np.nan

    isTakeoff = traj.data.isTakeoff.iloc[0]
    typecode = traj.data.typecode.iloc[0]

    if isTakeoff and (typecode in AC2CONSIDER):

        # Fuel for main engine start (MES), for engine and for APU
        MESengine = fuelMESengine(typecode, startupTime=DEFAULT_STARTUP_TIME, warmupTime=DEFAULT_WARMUP_TIME)
        MESapu = fuelMESapu(typecode, startupTime=DEFAULT_STARTUP_TIME, warmupTime=DEFAULT_WARMUP_TIME)

        # Taxi Fuel with engines running
        startTaxi = traj.data.startTaxi.iloc[0]
        lineupTime = traj.data.lineupTime.iloc[0]
        if (startTaxi is not np.nan) and (lineupTime is not np.nan):
            normTAXIengine = fuelTaxiEngine(typecode, startTaxi, lineupTime)

    
    # Write data to traj
    traj.data.loc[:, 'MESengine'] = MESengine
    traj.data.loc[:, 'MESapu'] = MESapu
    traj.data.loc[:, 'normTAXIengine'] = normTAXIengine
    

    return traj


def extAGPSTaxiFuel(traj, startupTime=DEFAULT_STARTUP_TIME, warmupTime=DEFAULT_WARMUP_TIME):
    ECSapu = np.nan
    extAGPStug = np.nan

    isTakeoff = traj.data.isTakeoff.iloc[0]
    typecode = traj.data.typecode.iloc[0]

    if isTakeoff and (typecode in AC2CONSIDER):
        startTaxi = traj.data.startTaxi.iloc[0]
        lineupTime = traj.data.lineupTime.iloc[0]
        taxiDuration = lineupTime - startTaxi

        # Calculate fuel required for ECS of APU during external AGPS operaitons
        if taxiDuration > timedelta(seconds=0):
            ECSapu = fuelECSapu(typecode, agpsDuration=taxiDuration)

    traj.data.loc[:, 'extAGPSapu'] = ECSapu
    traj.data.loc[:, 'extAGPStug'] = extAGPStug

    return traj




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



def dfAPUfuel() -> pd.DataFrame:
    """
    Creates and returns a DataFrame containing APU (Auxiliary Power Unit) fuel consumption data.

    The DataFrame includes details of APU fuel consumption for different aircraft types and operating modes.
    The data is based on ICAO Document 9889 Table 3-A1-5, which provides an advanced approach to calculate
    fuel consumption.

    Returns:
        pd.DataFrame: A DataFrame with columns representing various aircraft types and rows for different
                      APU operating modes, detailing the fuel consumption rates.
    """

    return DF_APU


def getAPUfuel(typecode: str, df_apu = DF_APU) -> pd.Series:
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

    # Return the selected row as a Series from the DataFrame
    return df_apu.iloc[idx]


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



def fuelMESengine(typecode: str, 
                  startupTime=DEFAULT_STARTUP_TIME, 
                  warmupTime=DEFAULT_WARMUP_TIME) -> float:
    
    """
    Calculates the total fuel consumption for main engine start (MES) based on the aircraft type and the number of engines.

    This function estimates the fuel consumption required to start all engines sequentially (i.e. one after another) for 
    a given aircraft type, considering both the run-up time needed to start each engine and an additional warm-up period 
    after the last engine starts. After an engine has started up, idle thrust conditions are assumed.

    Assumptions:
        * Engines are started sequentially, one after another.
        * Each engine start takes `startupTime` seconds.
        * There is no slack or delay between the start of successive engines.
        * After the start of the last engine, it warms up for `warmupTime` seconds.

    Args:
        typecode (str): The ICAO typecode of the aircraft.
        startupTime (int, optional): The time in seconds required to start each engine. Defaults to `DEFAULT_STARTUP_TIME`.
        warmupTime (int, optional): The warm-up time in seconds after all engines have started. Defaults to `DEFAULT_WARMUP_TIME`.

    Returns:
        float: The total fuel consumption for main engine start in kilograms.

    Notes:
        - The function first retrieves the number of engines and the idle fuel flow rate (idleFF) for the specified aircraft type
          using the `getNengine` and `getIdleFF` functions.
        - Fuel consumption is calculated based on the sum of fuel used for each engine start and warm-up period.
        - The `typecode` is converted to uppercase to ensure consistency in lookups.

    Example:
        >>> fuel_consumption = fuelMESengine('A320')
        >>> print(fuel_consumption)
        120.0  # Example output, actual value depends on data and typecode
    """

    typecode = typecode.upper()

    # Get number of engines and idle fuel flow for the aircraft type
    nEngine = getNengine(typecode)
    idleFF = getIdleFF(typecode)

    # Initialize total fuel consumption
    fuelConsumption = 0

    # Loop to calculate fuel consumption for each engine
    for i in range(1, nEngine + 1):
        fuelConsumption += idleFF * (i * startupTime + warmupTime)

    return fuelConsumption


def fuelMESapu(typecode: str, 
               df_apu=DF_APU, 
               startupTime=DEFAULT_STARTUP_TIME, 
               warmupTime=DEFAULT_WARMUP_TIME) -> float:
    
    """
    Calculates the fuel consumption of the Auxiliary Power Unit (APU) in kilograms (kg) for the main engine start 
    sequence based on the aircraft type.

    This function estimates the total APU fuel consumption required during the sequential startup of the aircraft's 
    engines (i.e. engines are started up one after another). It uses the type of aircraft to determine the number of 
    engines and their respective fuel consumption rates. The calculation accounts for both the run-up and warm-up 
    times specified for the startup process, adjusting fuel consumption rates from hours to seconds as necessary.

    Args:
        typecode (str): The ICAO typecode of the aircraft for which the APU fuel consumption is calculated.
        df_apu (pd.DataFrame): A DataFrame containing APU fuel consumption data for various aircraft types.
                               Defaults to `DF_APU`.
        startupTime (int, optional): The time in seconds for which each engine runs up during the startup. 
                                     Defaults to `DEFAULT_STARTUP_TIME`.
        warmupTime (int, optional): The time in seconds for which the last engine warms up after all engines are started.
                                    Defaults to `DEFAULT_WARMUP_TIME`.

    Returns:
        float: The total APU fuel consumption in kilograms (kg) for the main engine start sequence.

    Assumptions:
        * Engines are started sequentially.
        * Each engine start takes `startupTime` seconds.
        * There is no slack time between engine starts; engines start one immediately after the other.
        * After the start of the last engine, this engine warms up for `warmupTime` seconds.
        * APU fuel consumption rates (high and normal) are provided in kg/h and are converted to per-second rates 
          within the function.

    Example:
        >>> df_apu = getAPUfuel_df()
        >>> fuel_consumed = fuelMESapu('A320', df_apu, startupTime=150, warmupTime=180)
        >>> print(fuel_consumed)
        2.75  # Example output, actual value may differ based on data

    Notes:
        - Ensure that the DataFrame `df_apu` is correctly formatted and contains the necessary APU fuel consumption 
          rates for the specified `typecode`.
        - The function converts the `typecode` to uppercase to ensure consistency in lookups.
        - The function calculates fuel consumption by determining the fuel used during the startup time for each engine 
          and the warm-up period after all engines are started.

    """


    typecode = typecode.upper()

    apuFuel = getAPUfuel(typecode, df_apu)
    nEngine = getNengine(typecode)

    fuelConsumption = (nEngine * apuFuel['high'] / 3600 * startupTime) + (apuFuel['normal'] / 3600 * warmupTime)

    return fuelConsumption


def fuelTaxiEngine(typecode: str, 
                   startTaxi: datetime, 
                   lineupTime: datetime, 
                   singleEngine = False):
    
    """
    Calculates the fuel consumption of the engine(s) during the taxi phase for a specified aircraft type.

    This function estimates the total fuel consumption of the aircraft's engines during taxiing from the start time 
    (`startTaxi`) to the lineup time on a runway (`lineupTime`). The calculation is based on the aircraft type, the 
    number of engines, the idle fuel flow rate, and whether a single-engine taxi procedure is used.

    Args:
        typecode (str): The ICAO typecode of the aircraft.
        startTaxi (datetime): The timestamp when the taxi phase starts.
        lineupTime (datetime): The timestamp when the taxi phase ends (when the aircraft is lined up for takeoff).
        singleEngine (bool, optional): If True, assumes only 50% of the engines are running during taxi (e.g., 
                                       single-engine taxi for a two-engine aircraft). Defaults to False.

    Returns:
        float: The total fuel consumption in kilograms (kg) for the taxi phase.

    Notes:
        - The function retrieves the idle fuel flow rate (`idleFF`) and the number of engines (`nEngine`) for the 
          specified aircraft type using the `getIdleFF` and `getNengine` functions.
        - The fuel consumption is calculated by multiplying the taxi duration (in seconds) by the number of engines 
          and the idle fuel flow rate.
        - If `singleEngine` is True, the fuel consumption is halved, assuming only half of the engines are operating.
        - The `typecode` is converted to uppercase to ensure consistency in lookups.

    Example:
        >>> from datetime import datetime
        >>> fuel_consumption = fuelTaxiEngine('A320', datetime(2024, 6, 1, 12, 0, 0), datetime(2024, 6, 1, 12, 10, 0))
        >>> print(fuel_consumption)
        45.0  # Example output, actual value depends on data and typecode

    """


    typecode = typecode.upper()

    idleFF = getIdleFF(typecode)
    nEngine = getNengine(typecode)

    taxiDuration = lineupTime - startTaxi
    taxiDuration = taxiDuration.total_seconds()

    
    fuelConsumption = taxiDuration * nEngine * idleFF

    if singleEngine:
        fuelConsumption = fuelConsumption * 0.5

    return fuelConsumption


def fuelECSapu(typecode: str, 
               agpsDuration: datetime.timedelta,
               df_apu = DF_APU) -> float:
    """ 
    Calculates the fuel consumption of the Auxiliary Power Unit (APU) in kilograms (kg) for powering the Environmental 
    Control System (ECS) of the aircraft.

    This function estimates the total APU fuel consumption required when the APU is used to energize the ECS during 
    a specified duration. The fuel consumption is based on the normal fuel consumption rate of the APU for the given 
    aircraft type and the duration for which the ECS is powered.

    Args:
        typecode (str): The ICAO typecode of the aircraft.
        agpsDuration (datetime.timedelta): The duration for which the APU powers the ECS.
        df_apu (pd.DataFrame, optional): A DataFrame containing APU fuel consumption data for various aircraft types.
                                         Defaults to `DF_APU`.

    Returns:
        float: The total APU fuel consumption in kilograms (kg) for powering the ECS.

    Notes:
        - The function retrieves APU fuel consumption rates using the `getAPUfuel` function, which looks up data 
          based on the aircraft typecode.
        - The normal APU fuel consumption rate is provided in kilograms per hour (kg/h) and is converted to a 
          per-second rate within the function for accurate calculations.
        - The `typecode` is converted to uppercase to ensure consistency in lookups.

    Example:
        >>> from datetime import timedelta
        >>> fuel_consumed = fuelECSapu('A320', timedelta(hours=1))
        >>> print(fuel_consumed)
        5.0  # Example output, actual value may differ based on data

    """

    typecode = typecode.upper()

    apuFuel = getAPUfuel(typecode, df_apu)

    fuelConsumption = apuFuel['normal'] / 3600 * agpsDuration.total_seconds()

    return fuelConsumption



def plotTrajs2PDF(df_movements, 
                  gnd_trajs, 
                  nPages: int, 
                  trajsPerPage: int,
                  airport_str = 'LSZH',
                  rwys = [],
                  dir_path=os.getcwd(),
                  plotName='check_flights.pdf',
                  colors = ['r', 'b', 'g', 'y', 'c']):

    plot_path = dir_path + '/' + plotName

    if trajsPerPage > len(colors):
        trajsPerPage = len(colors)

    # Sample all flight_id once (n_Pages * trajsPerPage)
    total_sample_size = nPages * trajsPerPage

    if total_sample_size > len(df_movements):
        nPages = np.floor(len(df_movements)/trajsPerPage)
        total_sample_size = nPages * trajsPerPage

    sample = df_movements.sample(n=total_sample_size, replace=False)

    # Access gnd_trajs once (access is slow due to its size)
    trajs_tmp = gnd_trajs[sample.flight_id.values]

    # Split the sample into smaller chunks for each page
    sample_chunks = [sample[i:i + trajsPerPage] for i in range(0, len(sample), trajsPerPage)]

    with PdfPages(plot_path) as pdf:
        # Iterate over each chunk for plotting
        for repeat, sample_chunk in enumerate(sample_chunks):
            fig, ax = plt.subplots(1, 1, figsize=(12, 12), subplot_kw=dict(projection=EuroPP()))
            airports[airport_str].plot(ax, by='aeroway', aerodrome=dict(alpha=0))
            ax.spines["geo"].set_visible(False)
            ax.set_extent((8.5230, 8.5855, 47.4904, 47.4306))

            # Plot runway geometries once
            for rwy in rwys:
                ax.plot(*rwy.exterior.xy, transform=data_proj, color='m')

            # Plot the trajectories for the current chunk
            for i, (index, flt) in enumerate(sample_chunk.iterrows()):
                
                # Plot the entire trajectory
                trajs_tmp[flt.flight_id].plot(ax=ax, color=colors[i], label=f'{flt.flight_id} RWY {flt.takeoffRunway} d={round(flt.taxiDistance,2)}')

                # Highlight taxi part
                trajs_tmp[flt.flight_id].between(flt.startTaxi, flt.lineupTime).plot(ax=ax, color=colors[i], linewidth=4)

                # Mark pushback part if applicable
                if flt.isPushback:
                    trajs_tmp[flt.flight_id].between(flt.startPushback, flt.startTaxi).plot(ax=ax, color='m', linewidth=4)

            plt.legend(loc='upper right')
            pdf.savefig()  # Save the current page into the PDF
            plt.close()  # Close the plot to free memory

    return