import numpy
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense
from matplotlib import pyplot as plt
from tensorflow import keras
import numpy as np
import pandas as pd
from collections import Counter
import resampy
import matplotlib.pyplot as py
from sklearn.metrics import mean_squared_error, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from keras import optimizers

# Define file pool
busPoolTrain = ['./Dataset/bus/A_2017.11.30_16.48.26_output.csv', './Dataset/bus/A_2018.01.25_16.33.53_output.csv','./Dataset/bus/A_2018.01.25_17.27.30_output.csv','./Dataset/bus/A_2018.01.25_18.02.07_output.csv','./Dataset/bus/A_2018.01.25_19.50.40_output.csv','./Dataset/bus/A_2018.01.26_11.26.26_output.csv','./Dataset/bus/A_2018.01.27_10.58.49_output.csv','./Dataset/bus/A_2018.01.27_11.12.23_output.csv','./Dataset/bus/A_2018.01.27_12.10.00_output.csv', './Dataset/bus/B_2018.01.25_16.33.45_output.csv', './Dataset/bus/B_2018.01.25_17.27.27_output.csv', './Dataset/bus/B_2018.01.25_18.02.03_output.csv', './Dataset/bus/B_2018.01.25_19.50.43_output.csv', './Dataset/bus/B_2018.01.27_10.58.49_output.csv', './Dataset/bus/B_2018.01.27_11.12.22_output.csv', './Dataset/bus/B_2018.01.27_12.09.59_output.csv']
pedestrianPoolTrain = ['./Dataset/pedestrian/A_2017.11.21_15.03.50_output.csv', './Dataset/pedestrian/A_2017.11.21_15.35.33_output.csv', './Dataset/pedestrian/A_2017.11.21_17.35.54_output.csv', './Dataset/pedestrian/A_2017.11.22_07.57.13_output.csv', './Dataset/pedestrian/A_2017.11.29_18.40.47_output.csv', './Dataset/pedestrian/A_2017.11.30_08.18.06_output.csv', './Dataset/pedestrian/A_2017.11.30_11.59.09_output.csv', './Dataset/pedestrian/A_2017.11.30_11.59.10_output.csv', './Dataset/pedestrian/A_2017.11.30_11.59.10_output.csv', './Dataset/pedestrian/B_2017.12.14_17.49.58_output.csv', './Dataset/pedestrian/B_2017.12.17_13.53.32_output.csv', './Dataset/pedestrian/B_2017.12.17_14.55.18_output.csv','./Dataset/pedestrian/B_2018.01.19_16.06.32_output.csv','./Dataset/pedestrian/B_2018.02.11_13.30.46_output.csv']

# busPoolTest = ['./Dataset/bus/B_2018.01.25_16.33.45_output.csv', './Dataset/bus/B_2018.01.25_17.27.27_output.csv', './Dataset/bus/B_2018.01.25_18.02.03_output.csv', './Dataset/bus/B_2018.01.25_19.50.43_output.csv', './Dataset/bus/B_2018.01.27_10.58.49_output.csv', './Dataset/bus/B_2018.01.27_11.12.22_output.csv', './Dataset/bus/B_2018.01.27_12.09.59_output.csv']
# pedestrianPoolTest = ['./Dataset/pedestrian/B_2017.12.14_17.49.58_output.csv', './Dataset/pedestrian/B_2017.12.17_13.53.32_output.csv', './Dataset/pedestrian/B_2017.12.17_14.55.18_output.csv','./Dataset/pedestrian/B_2018.01.19_16.06.32_output.csv','./Dataset/pedestrian/B_2018.02.11_13.30.46_output.csv']


# Columns to read
# phy_ul_pucch_rssi ?
columns = ['ue_ident', 'timestamp', 'phy_ul_pucch_rssi']

# Load the data
# Bus
bus = []
for file in busPoolTrain:
    bus.append(pd.read_csv(file, usecols=columns, sep=';'))
# Pedestrian
pedestrian = []
for file in pedestrianPoolTrain:
    pedestrian.append(pd.read_csv(file, usecols=columns, sep=';'))

busFrame = pd.concat(bus, axis=0, ignore_index=True)
pedestrianFrame = pd.concat(pedestrian, axis=0, ignore_index=True)

busIDs, busCount = np.unique(busFrame['ue_ident'], return_counts=True)
pedestrianIDs, pedestrianCount = np.unique(pedestrianFrame['ue_ident'], return_counts=True)

stats = busFrame.append(pedestrianFrame)
print('lengths :', Counter(stats['ue_ident']).values())
print('median: ', numpy.median(list(Counter(stats['ue_ident']).values())))
print('avg: ', numpy.mean(list(Counter(stats['ue_ident']).values())))

#Counter(words).keys() # equals to list(set(words))
#(words).values() # counts the elements' frequency

# Cross validation of interpolation
#cut = [50, 1000, 2000, 3000, 3500, 4000, 4500, 5000, 5500, 6000]
# sizes = [2000, 3000, 3500, 4000, 4500, 5000, 5500, 6000]
cut = [2000]
sizes = [2000]

# define min max scaler
scaler = MinMaxScaler()

for c in cut:
    for s in sizes:
        X = []
        y = []
        for v in busIDs:
            measures = pd.DataFrame(busFrame.loc[busFrame['ue_ident'] == v])
            if len(measures) >= c:
                # Sort values
                measures.sort_values(['ue_ident', 'timestamp'], ascending=[True, True], inplace=True)
                # Convert to array
                measures = measures['phy_ul_pucch_rssi'].to_numpy()
                # Interpolation
                measures = resampy.resample(measures, len(measures), s, axis=-1)
                # transform data
                measures = scaler.fit_transform(measures.reshape(-1, 1))
                X.append(measures)
                y.append('bus')


        for v in pedestrianIDs:
            measures = pd.DataFrame(pedestrianFrame.loc[pedestrianFrame['ue_ident'] == v])
            if len(measures) >= c:
                # Sort values
                measures.sort_values(['ue_ident', 'timestamp'], ascending=[True, True], inplace=True)
                # Convert to array
                measures = measures['phy_ul_pucch_rssi'].to_numpy()
                # Interpolation
                measures = resampy.resample(measures, len(measures), s, axis=-1)
                # transform data
                measures = scaler.fit_transform(measures.reshape(-1, 1))
                X.append(measures)
                y.append('pedestrian')

        X = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)

        enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
        enc = enc.fit(y_train)
        y_train = enc.transform(y_train)
        y_test = enc.transform(y_test)

        # nv = 1
        # rate = float(10000000000000)
        # while nv == 1:
        # rate = rate/10

        # Define model architecture
        model = keras.Sequential()
        model.add(SimpleRNN(32, input_shape=[X_train.shape[1], X_train.shape[2]]))
        model.add(Dense(y_train.shape[1], activation='softmax'))

        # model = keras.Sequential()
        # model.add(keras.layers.LSTM(units=128, input_shape=[X_train.shape[1], X_train.shape[2]]))
        # model.add(keras.layers.Dropout(rate=0.5))
        # model.add(keras.layers.Dense(units=128, activation='softmax'))
        # model.add(keras.layers.Dense(y_train.shape[1], activation='softmax'))

        #opt = optimizers.Adam(lr=rate)
        #opt = optimizers.SGD(lr=rate, decay=1e-6, momentum=0.9, nesterov=True)

        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['acc']
        )

        history = model.fit(
            X_train, y_train,
            epochs=10,
            batch_size=32,
            validation_split=0.1,
            shuffle=False
        )

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

