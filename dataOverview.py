from Classes.dataReader import DataReader
from Static.typeEnum import Paths
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Columns to read
columns = ['ue_ident', 'timestamp', 'phy_ul_pucch_rssi', 'phy_ul_pucch_rssi']
patterns = [Paths.bus, Paths.car, Paths.pedestrian, Paths.static, Paths.train]

reader = DataReader(columns, patterns)

bus = reader.retrievePatternData(Paths.bus)
car = reader.retrievePatternData(Paths.car)
pedestrian = reader.retrievePatternData(Paths.pedestrian)
static = reader.retrievePatternData(Paths.static)
train = reader.retrievePatternData(Paths.train)

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
samplePedestrian = pedestrian[pedestrian['ue_ident'] == list(Counter(pedestrian['ue_ident']).keys())[2]]
sampleStatic = static[static['ue_ident'] == list(Counter(static['ue_ident']).keys())[0]]
sampleTrain = train[train['ue_ident'] == list(Counter(train['ue_ident']).keys())[0]]

samples = [sampleBus, sampleCar, samplePedestrian, sampleStatic, sampleTrain]
titles = ['bus', 'car', 'pedestrian', 'static', 'train']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

x = 'timestamp'
y = 'phy_ul_pucch_rssi'

for index, s in enumerate(samples):
    plt.scatter(s[x], s[y], c=colors[index])
    plt.ylabel(x)
    plt.xlabel(y)
    plt.title(titles[index])
    plt.show()

print()