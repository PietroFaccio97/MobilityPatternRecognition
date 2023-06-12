import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from Classes import dataShaper
from Classes.dataReader import DataReader
from Static.typeEnum import Paths

def retrieve_measures(time_series, labels):
    """
    Extracts the measures from each time series and appends them together.

    Args:
        time_series (ndarray): Array of time series data in the shape (n, 9, 53).
        labels: target values

    Returns:
        ndarray: Array of measures for each time series and metric, shape (n, num_measures * num_metrics).
    """
    measures = []
    target_values = []

    for index, series in enumerate(time_series):
        for measure in series:
            measures.append(measure)
            # Label the measures with the target class of the time series
            target_values.append(labels[index])

    return measures, target_values
def retrieve_standard_deviations(time_series, labels):
    """
    Calculates the standard deviation among the measures for each metric in a time series array.

    Args:
        time_series (ndarray): Array of time series data in the shape (n, 9, 53).
        labels: target values

    Returns:
        ndarray: Array of standard deviations for each time series and metric, shape (n, 53).
    """
    n, num_measures, num_metrics = time_series.shape
    standard_deviations = np.zeros((n, num_metrics))
    target_values = []

    for i in range(n):
        for metric in range(num_metrics):
            metric_data = time_series[i, :, metric]
            metric_std = np.std(metric_data)
            standard_deviations[i, metric] = metric_std
        # Label the standard deviation values with the target class of the time series
        target_values.append(labels[i])

    return standard_deviations, target_values

# Columns to read
core_features = ['ue_ident', 'timestamp']
#features = ['phy_ul_pusch_sinr', 'phy_ul_pucch_sinr', 'phy_ul_mcs', 'mac_dl_cqi_offset', 'mac_ul_snr_offset', 'phy_ul_pusch_rssi', 'phy_ul_pucch_rssi', 'phy_ul_pucch_ni', 'phy_ul_turbo_iters', 'phy_ul_n_samples', 'phy_ul_n_samples_pucch', 'phy_dl_mcs', 'phy_dl_n_samples']
features = ['mac_rnti', 'mac_dl_cqi', 'mac_dl_mcs', 'mac_dl_brate', 'mac_dl_ok', 'mac_dl_nok', 'phy_ul_pusch_sinr', 'phy_ul_pucch_sinr', 'phy_ul_mcs', 'mac_ul_brate', 'mac_ul_ok', 'mac_ul_nok', 'mac_ul_bsr', 'mac_pci', 'mac_nof_tti', 'mac_cc_idx', 'mac_dl_buffer', 'mac_dl_ri', 'mac_dl_pmi', 'mac_phr', 'mac_dl_cqi_offset', 'mac_ul_snr_offset', 'mac_ul_rssi', 'mac_fec_iters', 'mac_dl_mcs_samples', 'mac_ul_mcs', 'mac_ul_mcs_samples', 'phy_ul_n', 'phy_ul_pusch_rssi', 'phy_ul_pusch_tpc', 'phy_ul_pucch_rssi', 'phy_ul_pucch_ni', 'phy_ul_turbo_iters', 'phy_ul_n_samples', 'phy_ul_n_samples_pucch', 'phy_dl_mcs', 'phy_dl_pucch_tpc', 'phy_dl_n_samples', 'rf_o', 'rf_u', 'rf_l', 'rf_error', 'sys_process_realmem_kB', 'sys_process_virtualmem_kB', 'sys_process_realmem', 'sys_thread_count', 'sys_process_cpu_usage', 'sys_system_mem', 'sys_cpu_count', 'n_UEs', 'n_PRBs', 'power_pdu', 'id_ue'] # Remove only label feature
columns = core_features + features

# Ascending flag array for sorting
ascendingFlags = [True for _ in columns]
# Pattern classes paths
patterns = [Paths.bus, Paths.car, Paths.pedestrian, Paths.static, Paths.train]

#Tune thresholds for outliers cut off based on the feature
outlierThresholds = {'bus': {'phy_ul_pucch_rssi': 50,
                             'phy_ul_pusch_rssi': 50,
                             'phy_ul_pucch_sinr': 30,
                             'phy_ul_pusch_sinr': 30},
                     'car': {'phy_ul_pucch_rssi': 50,
                             'phy_ul_pusch_rssi': 50,
                             'phy_ul_pucch_sinr': 30,
                             'phy_ul_pusch_sinr': 30},
                     'pedestrian': {'phy_ul_pucch_rssi': 50,
                             'phy_ul_pusch_rssi': 50,
                             'phy_ul_pucch_sinr': 30,
                             'phy_ul_pusch_sinr': 30},
                     'static': {'phy_ul_pucch_rssi': 50,
                             'phy_ul_pusch_rssi': 50,
                             'phy_ul_pucch_sinr': 30,
                             'phy_ul_pusch_sinr': 30},
                     'train': {'phy_ul_pucch_rssi': 50,
                             'phy_ul_pusch_rssi': 50,
                             'phy_ul_pucch_sinr': 30,
                             'phy_ul_pusch_sinr': 30}}
#outliers = ['phy_ul_pucch_rssi', 'phy_ul_pusch_rssi', 'phy_ul_pucch_sinr', 'phy_ul_pusch_sinr']
outliers = ['phy_ul_pusch_sinr', 'phy_ul_pucch_sinr', 'phy_ul_mcs', 'mac_dl_cqi_offset', 'mac_ul_snr_offset', 'phy_ul_pusch_rssi', 'phy_ul_pucch_rssi', 'phy_ul_pucch_ni', 'phy_ul_turbo_iters', 'phy_ul_n_samples', 'phy_ul_n_samples_pucch', 'phy_dl_mcs', 'phy_dl_n_samples']

sequenceLength = 9

reader = DataReader(columns, patterns, outliers, outlierThresholds)

# Fetch data from csv files
bus = reader.retrievePatternData(Paths.bus)
car = reader.retrievePatternData(Paths.car)
pedestrian = reader.retrievePatternData(Paths.pedestrian)
static = reader.retrievePatternData(Paths.static)
train = reader.retrievePatternData(Paths.train)

# Order measures by time
bus = dataShaper.sort(data=bus, columnsOrder=columns, ascendingFlags=ascendingFlags)
car = dataShaper.sort(data=car, columnsOrder=columns, ascendingFlags=ascendingFlags)
pedestrian = dataShaper.sort(data=pedestrian, columnsOrder=columns, ascendingFlags=ascendingFlags)
static = dataShaper.sort(data=static, columnsOrder=columns, ascendingFlags=ascendingFlags)
train = dataShaper.sort(data=train, columnsOrder=columns, ascendingFlags=ascendingFlags)

# Retrieve time series from data
timeSeriesBus = dataShaper.retrieveSeries(data=bus, length=sequenceLength)
timeSeriesCar = dataShaper.retrieveSeries(data=car, length=sequenceLength)
timeSeriesPedestrian = dataShaper.retrieveSeries(data=pedestrian, length=sequenceLength)
timeSeriesStatic = dataShaper.retrieveSeries(data=static, length=sequenceLength)
timeSeriesTrain = dataShaper.retrieveSeries(data=train, length=sequenceLength)

# Create series label
Xb, yb = dataShaper.labelAndShapeSeries(timeSeriesBus, columns, 'bus')
Xc, yc = dataShaper.labelAndShapeSeries(timeSeriesCar, columns, 'car')
Xp, yp = dataShaper.labelAndShapeSeries(timeSeriesPedestrian, columns, 'pedestrian')
Xs, ys = dataShaper.labelAndShapeSeries(timeSeriesStatic, columns, 'static')
Xt, yt = dataShaper.labelAndShapeSeries(timeSeriesTrain, columns, 'train')

# Obtain the shortest length
l = min(len(Xb), len(Xc), len(Xp), len(Xs), len(Xt))

# Cut series length to the shortest
Xb = Xb[:l]
yb = yb[:l]
Xc = Xc[:l]
yc = yc[:l]
Xp = Xp[:l]
yp = yp[:l]
Xs = Xs[:l]
ys = ys[:l]
Xt = Xt[:l]
yt = yt[:l]

X = Xb + Xc + Xp + Xs + Xt
y = yb + yc + yp + ys + yt

X = np.asarray(X)
y = np.asarray(y).reshape(-1, 1)

#-----------------------------------------------RAW MEASURES-----------------------------------------------------------
decomposed_X, decomposed_y = retrieve_measures(X, y)

# Convert lists to numpy arrays
X_fit = np.array(decomposed_X)
y_fit = np.array(decomposed_y)

# Reshape y_fit to be a 1-dimensional array
y_fit = y_fit.ravel()

# Train a random forest classifier
model = RandomForestClassifier()
model.fit(X_fit, y_fit)

# Retrieve feature importance scores
feature_importance = model.feature_importances_

# Sort features based on importance scores
sorted_features = sorted(zip(feature_importance, range(len(feature_importance))), reverse=True)

# Extract feature importance and feature indices
sorted_importance, sorted_indices = zip(*sorted_features)

# Print feature importance scores
for importance, feature_index in sorted_features:
    feature_name = features[feature_index]
    print(f"Feature {feature_name}: Importance = {importance}")

# Plot the feature weights
plt.figure(figsize=(10, 6))
plt.bar(range(len(sorted_importance)), sorted_importance, align='center')
plt.xticks(range(len(sorted_importance)), [features[i] for i in sorted_indices], rotation=90)
plt.ylabel('Feature Weight')
plt.title('Feature Weights - Raw Measures')
plt.tight_layout()
plt.show()


#----------------------------------------------STANDARD DEVIATION------------------------------------------------------
sd_X, sd_y = retrieve_standard_deviations(X, y)

# Convert lists to numpy arrays
X_fit = np.array(sd_X)
y_fit = np.array(sd_y)

# Reshape y_fit to be a 1-dimensional array
y_fit = y_fit.ravel()

# Train a random forest classifier
model = RandomForestClassifier()
model.fit(X_fit, y_fit)

# Retrieve feature importance scores
feature_importance = model.feature_importances_

# Sort features based on importance scores
sorted_features = sorted(zip(feature_importance, range(len(feature_importance))), reverse=True)

# Extract feature importance and feature indices
sorted_importance, sorted_indices = zip(*sorted_features)

# Print feature importance scores
for importance, feature_index in sorted_features:
    feature_name = features[feature_index]
    print(f"Feature {feature_name}: Importance = {importance}")

# Plot the feature weights
plt.figure(figsize=(10, 6))
plt.bar(range(len(sorted_importance)), sorted_importance, align='center')
plt.xticks(range(len(sorted_importance)), [features[i] for i in sorted_indices], rotation=90)
plt.ylabel('Feature Weight')
plt.title('Feature Weights - Standard Deviation')
plt.tight_layout()
plt.show()