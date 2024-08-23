import traffic
from traffic.core import Traffic
import numpy as np
from datetime import timedelta
from shapely.geometry import base

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
    Detects takeoff events from a given trajectory and associates the takeoff with the correct runway.

    This function analyzes a flight trajectory to determine if a takeoff has occurred at a specified airport.
    If a takeoff is detected, the function identifies the runway used and marks the trajectory data with 
    the takeoff runway, lineup time, and a boolean flag indicating whether a takeoff occurred.

    Args:
        traj (Trajectory): A traffic flight object
        rwy_geometries (list of shapely.geometry.Polygon): A list of runway geometries (polygons) representing 
                                                           the airport's runways.
        df_rwys (pandas.DataFrame): from traffic.core import airports -> airports[airport_str].runways.data
        airport_str (str): The ICAO code of the airport to check for takeoff. Defaults to 'LSZH'.
        gsColName (str, optional): The name of the column in `traj` that contains ground speed data. Defaults to 'compute_gs'.

    Returns:
        Trajectory: The input trajectory with additional columns:
                    - 'lineupTime': The timestamp of the aircraft's lineup on the runway, or NaN if no takeoff was detected.
                    - 'takeoffRunway': The name of the detected takeoff runway, or an empty string if no takeoff was detected.
                    - 'isTakeoff': A boolean indicating whether a takeoff was detected.

    Notes:
        - The function assumes that the trajectory contains data relevant to the specified airport.

    Example:
        traj = takeoff_detection(traj, rwy_geometries, df_rwys, airport_str='LSZH', gsColName='compute_gs')
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
    Detects pushback events for a given flight trajectory using another method than the one specified in the traffic library.
    This detection is valid only for trajectories classified as takeoffs using method takeoff_detection() specified above.

    The function determines whether a pushback has occurred based on the flight's interaction with stand area geometries to be provided
    by the user. These geometries are polygons specifying the outset of the stand areas on an airport. The method calculates the pushback 
    duration, taxi duration, and marks the relevant timestamps in the trajectory data.

    Args:
        traj (Trajectory): A traffic flight object
        standAreas (list of shapely.geometry.Polygon): A list of stand area geometries (polygons) representing parking
                                                       or stand locations at the airport.
        airport_str (str): The ICAO code of the airport to check for pushback and taxi events. Defaults to 'LSZH'.

    Returns:
        Trajectory: The input trajectory with additional columns:
                    - 'isPushback': A boolean indicating whether a pushback was detected.
                    - 'startPushback': The timestamp when the pushback started, or NaN if not detected.
                    - 'startTaxi': The timestamp when taxiing started, or NaN if not detected.
                    - 'pushbackDuration': The duration of the pushback, or NaN if not detected.
                    - 'taxiDuration': The duration of the taxi, or NaN if not detected.

    Notes:
        - This function assumes that the trajectory corresponds to a takeoff event classified using method takeoff_detection().
        - It applies detection logic only if the takeoff flag is set in the trajectory data.

    Example:
        traj = alternative_pushback_detection(traj, standAreas, airport_str='LSZH')
    """

    lineupTime = traj.data.lineupTime.iloc[0]
    taxiDuration = np.nan
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

            # Taxi duration
            taxiDuration = lineupTime - startTaxi

            # Pushback duration
            pushbackDuration = startTaxi - startPushback

            # Check if taxiDuration is less than 0 seconds. This can happen if the ground coverage is not perfect and parts of the traj are missing.
            if taxiDuration < timedelta(seconds=0):
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
        
    # Write date to traj
    traj.data.loc[:, 'isPushback'] = isPushback
    traj.data.loc[:, 'startPushback'] = startPushback
    traj.data.loc[:, 'startTaxi'] = startTaxi
    traj.data.loc[:, 'pushbackDuration'] = pushbackDuration
    traj.data.loc[:, 'taxiDuration'] = taxiDuration

    return traj
