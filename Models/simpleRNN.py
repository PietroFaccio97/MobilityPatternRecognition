from Classes.dataReader import DataReader
from Classes import dataShaper
from Static.typeEnum import Paths

import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow import keras
from keras.layers import SimpleRNN, Dense
from sklearn.metrics import classification_report

# Columns to read
columns = ['ue_ident', 'timestamp', 'phy_ul_pucch_rssi', 'phy_ul_pucch_rssi']
ascendingFlags = [True, True, True, True]
patterns = [Paths.bus, Paths.car, Paths.pedestrian, Paths.static, Paths.train]

#Tune thresholds for outliers cut off based on the feature
outlierThresholds = {'bus': {'phy_ul_pucch_rssi': 50}, 'car': {'phy_ul_pucch_rssi': 50}, 'pedestrian': {'phy_ul_pucch_rssi': 50}, 'static': {'phy_ul_pucch_rssi': 50}, 'train': {'phy_ul_pucch_rssi': 50}}
outliers = ['phy_ul_pucch_rssi']

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
timeSeriesBus = dataShaper.retrieveSeries(data=bus, length=10)
timeSeriesCar = dataShaper.retrieveSeries(data=car, length=10)
timeSeriesPedestrian = dataShaper.retrieveSeries(data=pedestrian, length=10)
timeSeriesStatic = dataShaper.retrieveSeries(data=static, length=10)
timeSeriesTrain = dataShaper.retrieveSeries(data=train, length=10)

# TODO
#  -input_shape[none, 1] -> no need to specify the squences length
#  -turn to deep RNN -> add layers
# Create series label
Xb, yb = dataShaper.labelAndShapeSeries(timeSeriesBus, 'bus')
Xc, yc = dataShaper.labelAndShapeSeries(timeSeriesCar, 'car')
Xp, yp = dataShaper.labelAndShapeSeries(timeSeriesPedestrian, 'pedestrian')
Xs, ys = dataShaper.labelAndShapeSeries(timeSeriesStatic, 'static')
Xt, yt = dataShaper.labelAndShapeSeries(timeSeriesTrain, 'train')

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
model.add(SimpleRNN(20, input_shape=[None, 1]))
model.add(Dense(y_train.shape[1], activation='softmax'))

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