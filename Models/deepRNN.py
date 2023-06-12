from Classes.dataReader import DataReader
from Classes import dataShaper
from Static.typeEnum import Paths

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow import keras
from keras.layers import SimpleRNN, Dense
from sklearn.metrics import classification_report

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

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle=True)

# Encode labels
enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
enc = enc.fit(y_train)
y_train = enc.transform(y_train)
y_test = enc.transform(y_test)

# Define model architecture
model = keras.Sequential()
model.add(SimpleRNN(20, return_sequences=True, input_shape=[None, len(features)]))
model.add(SimpleRNN(20, return_sequences=True))
#model.add(SimpleRNN(20, return_sequences=True))
#model.add(SimpleRNN(20, return_sequences=True))
model.add(SimpleRNN(20))
model.add(Dense(5, activation='softmax'))

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['acc']
)

history = model.fit(
    X_train, y_train,
    epochs=100,
    validation_split=0.1,
)

# list all data in history
print(history.history.keys())

y_pred = model.predict(X_test)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc'], loc='upper left')
plt.show()

val_pred = np.argmax(y_pred, axis=1)
val_test = np.argmax(y_test, axis=1)
print(classification_report(val_test, val_pred))
print()