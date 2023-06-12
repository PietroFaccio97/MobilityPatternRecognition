from Classes.dataReader import DataReader
from Classes import dataShaper
from Static.typeEnum import Paths

import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Columns to read
columns = ['ue_ident', 'timestamp', 'phy_ul_pucch_rssi', 'phy_ul_pucch_rssi']
ascendingFlags = [True, True, True, True]
patterns = [Paths.bus, Paths.car, Paths.pedestrian, Paths.static, Paths.train]

#Tune thresholds for outliers cut off based on the feature
outlierThresholds = {'bus': {'phy_ul_pucch_rssi': 50}, 'car': {'phy_ul_pucch_rssi': 50}, 'pedestrian': {'phy_ul_pucch_rssi': 50}, 'static': {'phy_ul_pucch_rssi': 50}, 'train': {'phy_ul_pucch_rssi': 50}}
outliers = ['phy_ul_pucch_rssi']

reader = DataReader(columns, patterns, outliers, outlierThresholds)

bus = reader.retrievePatternData(Paths.bus)
car = reader.retrievePatternData(Paths.car)
pedestrian = reader.retrievePatternData(Paths.pedestrian)
static = reader.retrievePatternData(Paths.static)
train = reader.retrievePatternData(Paths.train)

bus = dataShaper.sort(data=bus, columnsOrder=columns, ascendingFlags=ascendingFlags)
car = dataShaper.sort(data=car, columnsOrder=columns, ascendingFlags=ascendingFlags)
pedestrian = dataShaper.sort(data=pedestrian, columnsOrder=columns, ascendingFlags=ascendingFlags)
static = dataShaper.sort(data=static, columnsOrder=columns, ascendingFlags=ascendingFlags)
train = dataShaper.sort(data=train, columnsOrder=columns, ascendingFlags=ascendingFlags)

# Plot number of RAW elements in series
busSize = len(Counter(bus['ue_ident']).keys())
carSize = len(Counter(car['ue_ident']).keys())
pedestrianSize = len(Counter(pedestrian['ue_ident']).keys())
staticSize = len(Counter(static['ue_ident']).keys())
trainSize = len(Counter(train['ue_ident']).keys())

objects = ('bus', 'car', 'pedestrian', 'static', 'train')
y_pos = np.arange(len(objects))
sizes = [busSize, carSize, pedestrianSize, staticSize, trainSize]

plt.bar(y_pos, sizes, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Number of devices tracked')
plt.title('Pattern')
plt.show()

# Plot samples
sampleBus = bus[bus['ue_ident'] == list(Counter(bus['ue_ident']).keys())[0]]
sampleCar = car[car['ue_ident'] == list(Counter(car['ue_ident']).keys())[0]]
samplePedestrian = pedestrian[pedestrian['ue_ident'] == list(Counter(pedestrian['ue_ident']).keys())[0]]
sampleStatic = static[static['ue_ident'] == list(Counter(static['ue_ident']).keys())[0]]
sampleTrain = train[train['ue_ident'] == list(Counter(train['ue_ident']).keys())[0]]

samples = [sampleBus, sampleCar, samplePedestrian, sampleStatic, sampleTrain]
#samples = [bus, car, pedestrian, static, train]
titles = ['bus', 'car', 'pedestrian', 'static', 'train']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

x = 'timestamp'
y = 'phy_ul_pucch_rssi'

for index, s in enumerate(samples):
    plt.scatter(s[x], s[y], c=colors[index])
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(titles[index])
    plt.show()

# Retrieve time series from data
timeSeriesBus = dataShaper.retrieveSeries(data=bus, length=10)
timeSeriesCar = dataShaper.retrieveSeries(data=car, length=10)
timeSeriesPedestrian = dataShaper.retrieveSeries(data=pedestrian, length=10)
timeSeriesStatic = dataShaper.retrieveSeries(data=static, length=10)
timeSeriesTrain = dataShaper.retrieveSeries(data=train, length=10)

busSize = len(timeSeriesBus)
carSize = len(timeSeriesCar)
pedestrianSize = len(timeSeriesPedestrian)
staticSize = len(timeSeriesStatic)
trainSize = len(timeSeriesTrain)

objects = ('bus', 'car', 'pedestrian', 'static', 'train')
y_pos = np.arange(len(objects))
sizes = [busSize, carSize, pedestrianSize, staticSize, trainSize]

plt.bar(y_pos, sizes, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Number of series extracted')
plt.title('Pattern')
plt.show()