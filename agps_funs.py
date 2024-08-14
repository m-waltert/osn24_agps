import traffic
from traffic.core import Traffic
import numpy as np
from datetime import timedelta


def takeoff_taxi_detection(traj, 
                           rwy_geometries, 
                           df_rwys, 
                           airport_str,
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
    taxiDuration = np.nan
    lineupTime = np.nan
    pushbackDuration = np.nan
    startPushback = np.nan
    startTaxi = np.nan
    isTakeoff = False
    isPushback = False

    if traj.takeoff_from(airport_str):

        for i, rwy in enumerate(rwy_geometries):
            if traj.intersects(rwy):

                # Clip traj to runway geometry
                clipped_traj = traj.clip(rwy)

                if clipped_traj is None or clipped_traj.data.empty:
                    continue


                # if clipped_traj is None or clipped_traj.data.empty:
                #     continue

                # Check whether there is a "hole" in the traj, which can happen for runway 16 departures, that cross runway 10/28 (and the software
                # then wrongfully assumes that it is runway 10/28 departure)
                time_diff = traj.inside_bbox(rwy).data['timestamp'].diff().dt.total_seconds()
                if time_diff.dropna().max() > maxHoleDuration:
                    continue

                # Cache traj snippets
                first_5sec = clipped_traj.first(seconds=5).data
                last_5sec = clipped_traj.last(seconds=5).data
                last_20sec = clipped_traj.last(seconds=20).data
                last_60min_data = traj.last(minutes=60)

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

        if isTakeoff:
            # Check if Pushback executed?
            pushback = last_60min_data.pushback(airport_str)
            if pushback is not None:
                isPushback = True
                startPushback = pushback.start
                startTaxi = pushback.stop
                pushbackDuration = startTaxi - startPushback
            else:
                # Check if parking position exists & aircraft is longer on position than 30 seconds
                parkingPosition = last_60min_data.on_parking_position(airport_str).max()
                if (parkingPosition is not None) and (parkingPosition.duration > timedelta(seconds=30)):
                    startTaxi = parkingPosition.stop
                else:
                    startTaxi = last_60min_data.start if not last_60min_data.data.empty else np.nan

            if (startTaxi is not np.nan) and (lineupTime is not np.nan):
                taxiDuration = lineupTime - startTaxi


    traj.data.loc[:, 'isPushback'] = isPushback
    traj.data.loc[:, 'startPushback'] = startPushback
    traj.data.loc[:, 'startTaxi'] = startTaxi
    traj.data.loc[:, 'pushbackDuration'] = pushbackDuration
    traj.data.loc[:, 'lineupTime'] = lineupTime
    traj.data.loc[:, 'taxiDuration'] = taxiDuration
    traj.data.loc[:, 'takeoffRunway'] = takeoffRunway

    return traj


def alternative_pushback_detection(traj, standAreas):
    """ Alternative means of detecting a pushback. Only valid for trajs which show negativ taxi duration."""

    taxiDuration = traj.data.taxiDuration.iloc[0]
    startPushback = traj.data.startPushback.iloc[0]
    startTaxi = traj.data.startTaxi.iloc[0]
    lineupTime = traj.data.lineupTime.iloc[0]
    isPushback = False

    if taxiDuration < timedelta(seconds=0):
        for standArea in standAreas:
            clipped_traj = traj.clip(standArea)

            # Check whether traj is inside stand_area
            if clipped_traj is None or clipped_traj.data.empty:
                        continue


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

            isPushback = True

            # If start and end of pushback are identical
            if startPushback == startTaxi:
                startPushback = startPushback - timedelta(seconds=2)

            # Recalculate taxiDuration
            if (startTaxi is not np.nan) and (lineupTime is not np.nan):
                taxiDuration = lineupTime - startTaxi


        # If it is not a pushback, assume that taxi starts at begin of traj
        if not isPushback:
            startPushback = traj.start
            startTaxi = traj.start
            taxiDuration = lineupTime - startTaxi
        
    # Write date to traj
    traj.data.loc[:, 'isPushback'] = isPushback
    traj.data.loc[:, 'startPushback'] = startPushback
    traj.data.loc[:, 'startTaxi'] = startTaxi
    traj.data.loc[:, 'taxiDuration'] = taxiDuration

    return traj