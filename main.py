from math import radians, cos, sin, asin, sqrt

import sys
from xml.dom import minidom
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
from pykalman import KalmanFilter
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter


def get_data(file):
    xml = minidom.parse(file)
    trkpts = xml.getElementsByTagName("trkpt")
    lats = []
    lons = []
    datetimes = []

    for trkpt in trkpts:
        # Latitude
        lat = float(trkpt.attributes["lat"].value)
        # Longitude
        lon = float(trkpt.attributes["lon"].value)
        lats.append(lat)
        lons.append(lon)
        # Check for time element
        time_elems = trkpt.getElementsByTagName("time")
        if time_elems:
            t = time_elems[0].firstChild.nodeValue
            datetimes.append(t)
        else:
            # Append a default or null value if time is not present
            datetimes.append(None)

    columns = ['datetime', 'lat', 'lon']
    df = pd.DataFrame(list(zip(datetimes, lats, lons)), columns=columns)
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True, errors='coerce')
    return df

def output_gpx(points, output_filename):
    """
    Output a GPX file with latitude and longitude from the points DataFrame.
    """
    from xml.dom.minidom import getDOMImplementation

    def append_trkpt(pt, trkseg, doc):
        trkpt = doc.createElement('trkpt')
        trkpt.setAttribute('lat', '%.7f' % (pt['lat']))
        trkpt.setAttribute('lon', '%.7f' % (pt['lon']))
        trkseg.appendChild(trkpt)

    doc = getDOMImplementation().createDocument(None, 'gpx', None)
    trk = doc.createElement('trk')
    doc.documentElement.appendChild(trk)
    trkseg = doc.createElement('trkseg')
    trk.appendChild(trkseg)

    points.apply(append_trkpt, axis=1, trkseg=trkseg, doc=doc)

    with open(output_filename, 'w') as fh:
        doc.writexml(fh, indent=' ')

# Adapted from https://stackoverflow.com/questions/27928/calculate-distance-between-two-latitude-longitude-points-haversine-formula/21623206
def haversine(lat1, lon1, lat2, lon2):
    r = 6371000  # metres
    p = np.pi / 180

    a = 0.5 - np.cos((lat2-lat1)*p)/2 + np.cos(lat1*p) * \
        np.cos(lat2*p) * (1-np.cos((lon2-lon1)*p))/2
    return 2 * r * np.arcsin(np.sqrt(a))


def distance(df):

    df['dist'] = haversine(
        df['lat'].shift(), df['lon'].shift(), df['lat'], df['lon'])

    total_distance = df['dist'].sum()
    return total_distance

def kalman_smoothing(df):
    kalman_data = df[['lat', 'lon']]

    # Get initial state aka first row
    initial_state = kalman_data.iloc[0]
    observation_covariance = np.diag(
        [0.1, 0.1]) ** 2  # TODO: shouldn't be zero
    transition_covariance = np.diag(
        [0.1, 0.1]) ** 2  # TODO: shouldn't be zero
    transition = [[1, 0], [0, 1]]
                  # TODO: shouldn't (all) be zero

    kf = KalmanFilter(
        initial_state_mean=initial_state,
        initial_state_covariance=observation_covariance,
        observation_covariance=observation_covariance,
        transition_covariance=transition_covariance,
        transition_matrices=transition
    )

    kalman_smoothed, _ = kf.smooth(kalman_data)

    df = pd.DataFrame(kalman_smoothed, columns=['lat', 'lon'])

    return df

def delete_every_other_point(points):
    return points.iloc[::2]

def lowess_smoothing(df, frac=0.1):
    numerical_index = np.arange(len(df))
    clean_df = df.dropna(subset=['lat', 'lon'])
    # Apply LOWESS smoothing using the numerical index
    smoothed_lat = lowess(clean_df['lat'], numerical_index, frac=frac)[:, 1]
    smoothed_lon = lowess(clean_df['lon'], numerical_index, frac=frac)[:, 1]
    return pd.DataFrame({'lat': smoothed_lat, 'lon': smoothed_lon}, index=clean_df.index)
    
# Additional smoothing function using Simple Moving Average (SMA)
def sma_smoothing(df, window=5):
    sma_lat = df['lat'].rolling(window=window, min_periods=1).mean()
    sma_lon = df['lon'].rolling(window=window, min_periods=1).mean()
    return pd.DataFrame({'lat': sma_lat, 'lon': sma_lon}, index=df.index)
    
def gaussian_smoothing(df, sigma=2):
    smoothed_lat = gaussian_filter(df['lat'], sigma=sigma)
    smoothed_lon = gaussian_filter(df['lon'], sigma=sigma)
    return pd.DataFrame({'lat': smoothed_lat, 'lon': smoothed_lon}, index=df.index)
    
def savgol_smoothing(df, window_size=5, polyorder=2):
    # Ensure window size is odd and greater than polyorder
    if window_size % 2 == 0:
        window_size += 1

    smoothed_lat = savgol_filter(df['lat'], window_size, polyorder)
    smoothed_lon = savgol_filter(df['lon'], window_size, polyorder)
    return pd.DataFrame({'lat': smoothed_lat, 'lon': smoothed_lon}, index=df.index)


def main():
    input_gpx = sys.argv[1]
    dataset = input_gpx.split('data/')[1]

    dpoints = get_data(input_gpx).set_index('datetime')

    points = delete_every_other_point(dpoints)

    # Calculate unfiltered distance
    unfiltered_dist = distance(points)
    print(f'Unfiltered distance: {unfiltered_dist:.2f} m')

    # Apply Kalman smoothing and calculate distance
    kalman_smoothed_points = kalman_smoothing(points)
    kalman_smoothed_dist = distance(kalman_smoothed_points)
    print(f'Kalman Filtered distance: {kalman_smoothed_dist:.2f} m')

    # Apply LOWESS smoothing and calculate distance
    lowess_smoothed_points = lowess_smoothing(points)
    lowess_smoothed_dist = distance(lowess_smoothed_points)
    print(f'LOWESS Filtered distance: {lowess_smoothed_dist:.2f} m')
    
    # Apply SMA smoothing and calculate distance
    sma_smoothed_points = sma_smoothing(points)
    sma_smoothed_dist = distance(sma_smoothed_points)
    print(f'SMA Filtered distance: {sma_smoothed_dist:.2f} m')
    
    # Apply Gaussian smoothing and calculate distance
    gaussian_smoothed_points = gaussian_smoothing(points)
    gaussian_smoothed_dist = distance(gaussian_smoothed_points)
    print(f'Gaussian Filtered distance: {gaussian_smoothed_dist:.2f} m')
    
    # Apply Savitzky-Golay smoothing and calculate distance
    savgol_smoothed_points = savgol_smoothing(points)
    savgol_smoothed_dist = distance(savgol_smoothed_points)
    print(f'Savitzky-Golay Filtered distance: {savgol_smoothed_dist:.2f} m')

    # Output results into their appropriate folder
    if dataset == 'garibaldi.gpx':
        output_gpx(kalman_smoothed_points.reset_index(), './Garibaldi Result/kalman_out')
        output_gpx(lowess_smoothed_points.reset_index(), './Garibaldi Result/lowess_out')
        output_gpx(sma_smoothed_points.reset_index(), './Garibaldi Result/sma_out')
        output_gpx(gaussian_smoothed_points.reset_index(), './Garibaldi Result/gaussian_out')
        output_gpx(savgol_smoothed_points.reset_index(), './Garibaldi Result/savgol_out')
    elif dataset == 'upperFalls.gpx':
        output_gpx(kalman_smoothed_points.reset_index(), './Upperfalls Result/kalman_out')
        output_gpx(lowess_smoothed_points.reset_index(), './Upperfalls Result/lowess_out')
        output_gpx(sma_smoothed_points.reset_index(), './Upperfalls Result/sma_out')
        output_gpx(gaussian_smoothed_points.reset_index(), './Upperfalls Result/gaussian_out')
        output_gpx(savgol_smoothed_points.reset_index(), './Upperfalls Result/savgol_out')



if __name__ == '__main__':
    main()
