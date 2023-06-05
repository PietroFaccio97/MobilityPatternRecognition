import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from Classes import dataShaper
from Classes.dataReader import DataReader
from Static.typeEnum import Paths

# Columns to read
core_features = ['ue_ident', 'timestamp']
#features = ['ue_ident', 'timestamp', 'phy_ul_pucch_rssi', 'phy_ul_pusch_rssi', 'phy_ul_pucch_sinr', 'phy_ul_pusch_sinr']
features = ['phy_ul_pusch_sinr', 'phy_ul_pucch_sinr', 'phy_ul_mcs', 'mac_dl_cqi_offset', 'mac_ul_snr_offset', 'phy_ul_pusch_rssi', 'phy_ul_pucch_rssi', 'phy_ul_pucch_ni', 'phy_ul_turbo_iters', 'phy_ul_n_samples', 'phy_ul_n_samples_pucch', 'phy_dl_mcs', 'phy_dl_n_samples']
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

# Create train data for the model
decomposed_X = []
decomposed_y = []
#TODO use some particular metric instead of flat measures?
# e.g. calculate the mean and deviation of each feature measure among series
# Iterates over time series to append each single measure
for index, series in enumerate(X):
    for measure in series:
        decomposed_X.append(measure)
        # Label the measures with the target class of the time series
        decomposed_y.append(y[index])

# Convert lists to numpy arrays
X_fit = np.array(decomposed_X)
y_fit = np.array(decomposed_y)

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
plt.title('Feature Weights')
plt.tight_layout()
plt.show()
