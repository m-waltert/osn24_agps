import traffic
from traffic.core import Traffic
import numpy as np
from datetime import timedelta


def takeoff_detection(traj, 
                      rwy_geometries, 
                      df_rwys, 
                      airport_str='LSZH',
                      maxHoleDuration=60):
    """
    Detects takeoff and taxiing events from trajectory data, identifies the takeoff runway, and computes the taxi duration.

    This function analyzes a given aircraft trajectory to determine if a takeoff occurred. It checks for intersections with
    runway geometries and calculates key metrics such as lineup time, taxi start time, taxi duration, and the takeoff runway 
    designation. The results are stored back into the trajectory's data.

    Args:
        traj (Trajectory): The flight object representing the aircraft's movement data.
        rwy_geometries (list of Shapely geometries): A list of runway geometries to check for intersections with the trajectory.
        df_rwys (DataFrame): A DataFrame containing runway information, including bearing and runway names, i.e., airports[airport_str].runways.data
        airport_str (str): The ICAO code of the airport where the takeoff is being detected.
        maxHoleDuration (int): Maximum duration of a hole in the trajectory permitted, is provided in units seconds

    Returns:
        Trajectory: The input trajectory object with additional attributes for taxi start time, lineup time, taxi duration, 
        and the identified takeoff runway.

    Attributes added to traj.data:
        startTaxi (Timestamp or NaN): The timestamp of when taxiing started, or NaN if not detected.
        lineupTime (Timestamp or NaN): The timestamp of when the aircraft lined up on the runway, or NaN if not detected.
        taxiDuration (Timedelta or NaN): The duration of the taxi, calculated as the difference between lineup time and taxi start.
        takeoff_runway (str or NaN): The name of the runway from which the aircraft took off, or NaN if not detected.

    Raises:
        None

    Example:
        ```python
        updated_traj = takeoff_taxi_detection(traj, rwy_geometries, df_rwys, "LSZH")
        ```
    """

    takeoffRunway = ''
    lineupTime = np.nan
    isTakeoff = False

    if traj.takeoff_from(airport_str):

        for i, rwy in enumerate(rwy_geometries):
            if traj.intersects(rwy):

                # Clip traj to runway geometry
                clipped_traj = traj.clip(rwy)

                if (clipped_traj is None) or (clipped_traj.data.empty): #or (clipped_traj.duration < timedelta(seconds=60)):
                    continue

                # Check whether there is a "hole" in the traj, which can happen for runway 16 departures, that cross runway 10/28 (and the software
                # then wrongfully assumes that it is runway 10/28 departure)
                time_diff = traj.inside_bbox(rwy).data['timestamp'].diff().dt.total_seconds()
                if time_diff.dropna().max() > maxHoleDuration:
                    continue

                # Cache traj snippets
                first_5sec = clipped_traj.first(seconds=5).data
                last_5sec = clipped_traj.last(seconds=5).data
                last_20sec = clipped_traj.last(seconds=20).data
                #last_60min_data = traj.last(minutes=60)

                # Calculate ground speed and vertical rate
                median_gs = first_5sec.compute_gs.median() if not first_5sec.compute_gs.isna().all() else np.nan
                median_rate = np.nanmedian(last_5sec.vertical_rate) if not last_5sec.vertical_rate.isna().all() else np.nan

                if (median_gs < 30) and (median_rate > 500) and not last_20sec.empty:
                    isTakeoff = True

                    # Mean track during take-off
                    median_track = last_20sec.track.median()

                    # Line-up time
                    lineupTime = clipped_traj.start

                    # Find the takeoff runway
                    runwayBearings = df_rwys.iloc[2*i:2*i+2].bearing
                    idx = (runwayBearings - median_track).abs().idxmin()
                    takeoffRunway = df_rwys.name.loc[idx]

                    break

        # if isTakeoff:
        #     # Check if Pushback executed?
        #     pushback = last_60min_data.pushback(airport_str)
        #     if pushback is not None:
        #         isPushback = True
        #         startPushback = pushback.start
        #         startTaxi = pushback.stop
        #         pushbackDuration = startTaxi - startPushback
        #     else:
        #         # Check if parking position exists & aircraft is longer on position than 30 seconds
        #         parkingPosition = last_60min_data.on_parking_position(airport_str).max()
        #         if (parkingPosition is not None) and (parkingPosition.duration > timedelta(seconds=30)):
        #             startTaxi = parkingPosition.stop
        #         else:
        #             startTaxi = last_60min_data.start if not last_60min_data.data.empty else np.nan

        #     if (startTaxi is not np.nan) and (lineupTime is not np.nan):
        #         taxiDuration = lineupTime - startTaxi


    traj.data.loc[:, 'lineupTime'] = lineupTime
    traj.data.loc[:, 'takeoffRunway'] = takeoffRunway
    traj.data.loc[:, 'isTakeoff'] = isTakeoff

    return traj


def alternative_pushback_detection(traj, standAreas, airport_str='LSZH'):
    """ Alternative means of detecting a pushback. Only valid for trajs which show negativ taxi duration."""

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


            # # Check if taxiDuration is less than 0 seconds. This can happen if the ground coverage is not perfect and parts of the traj are missing.
            # if taxiDuration < timedelta(seconds=0):
            #     diffGS = df.compute_gs.diff()
            #     startTaxiIndex = leaveStandTimeIndex
            #     while startTaxiIndex < df.index.max() and diffGS.loc[startTaxiIndex + 1]:
            #         startTaxiIndex += 1

            #     startTaxi = df.loc[startTaxiIndex, 'timestamp']
            #     taxiDuration = lineupTime - startTaxi
            #     startPushback = startTaxi - timedelta(seconds=2)

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
