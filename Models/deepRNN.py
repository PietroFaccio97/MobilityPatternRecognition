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

from sklearn.metrics import confusion_matrix
import seaborn as sns

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

# outliers = ['phy_ul_pucch_rssi', 'phy_ul_pucch_sinr', 'phy_ul_pusch_rssi', 'phy_ul_pusch_sinr']
outliers = ['phy_ul_pucch_rssi', 'phy_ul_pucch_sinr', 'phy_ul_pusch_rssi', 'phy_dl_n_samples', 'phy_ul_pusch_sinr', 'phy_ul_pucch_ni', 'phy_ul_n_samples']

# Pattern classes paths
patterns = [Paths.bus, Paths.car, Paths.pedestrian, Paths.static, Paths.train]

# Columns to read
core_features = ['ue_ident', 'timestamp']
# Manually selected
features1 = ['phy_ul_pusch_sinr', 'phy_ul_pucch_sinr', 'phy_ul_mcs', 'mac_dl_cqi_offset', 'mac_ul_snr_offset', 'phy_ul_pusch_rssi', 'phy_ul_pucch_rssi', 'phy_ul_pucch_ni', 'phy_ul_turbo_iters', 'phy_ul_n_samples', 'phy_ul_n_samples_pucch', 'phy_dl_mcs', 'phy_dl_n_samples']
# From std
features2 = ['phy_ul_pucch_rssi', 'mac_dl_brate', 'phy_ul_pucch_sinr', 'phy_ul_pucch_ni', 'mac_dl_ok', 'phy_ul_pusch_rssi', 'phy_ul_pusch_sinr', 'mac_dl_nok', 'mac_dl_cqi_offset', 'mac_ul_ok', 'mac_ul_brate', 'mac_ul_snr_offset', 'mac_dl_cqi', 'mac_ul_nok', 'mac_phr']
# All phy
features3 = ['phy_ul_pucch_rssi', 'phy_dl_n_samples', 'phy_ul_pucch_sinr', 'phy_ul_pucch_ni', 'phy_dl_mcs', 'phy_ul_pusch_rssi', 'phy_ul_pusch_sinr', 'phy_ul_n_samples_pucch', 'phy_ul_n_samples', 'phy_ul_mcs', 'phy_ul_turbo_iters']
features4 = ['phy_ul_pucch_rssi', 'phy_dl_n_samples', 'phy_ul_pucch_sinr', 'phy_ul_pucch_ni', 'phy_dl_mcs', 'phy_ul_pusch_rssi', 'phy_ul_pusch_sinr', 'phy_ul_n_samples_pucch', 'phy_ul_n_samples', 'phy_ul_mcs', 'phy_ul_turbo_iters', 'mac_dl_ok', 'mac_dl_nok', 'mac_ul_ok', 'mac_dl_cqi', 'mac_ul_nok', 'mac_phr']
features5 = ['phy_ul_pucch_rssi', 'phy_ul_pucch_sinr', 'phy_ul_pucch_ni', 'phy_ul_pusch_rssi', 'phy_ul_pusch_sinr', 'mac_dl_ok', 'mac_dl_nok', 'mac_dl_cqi', 'mac_ul_ok', 'mac_ul_nok', 'mac_phr', 'phy_dl_n_samples', 'phy_ul_n_samples']
features6 = ['phy_ul_pucch_rssi', 'phy_ul_pucch_sinr', 'phy_ul_pusch_rssi', 'phy_dl_n_samples', 'phy_ul_pusch_sinr', 'phy_ul_pucch_ni', 'phy_ul_n_samples']

#features = [features1, features2, features3, features4, features5]
features = [features6]
sequenceLength = [5]
r_nn_layers = [5]

for i, f in enumerate(features):
    columns = core_features + f
    reader = DataReader(columns, patterns, outliers, outlierThresholds)

    # Fetch data from csv files
    bus = reader.retrievePatternData(Paths.bus)
    car = reader.retrievePatternData(Paths.car)
    pedestrian = reader.retrievePatternData(Paths.pedestrian)
    static = reader.retrievePatternData(Paths.static)
    train = reader.retrievePatternData(Paths.train)

    # Ascending flag array for sorting
    ascendingFlags = [True for _ in columns]
    # Order measures by time
    bus = dataShaper.sort(data=bus, columnsOrder=columns, ascendingFlags=ascendingFlags)
    car = dataShaper.sort(data=car, columnsOrder=columns, ascendingFlags=ascendingFlags)
    pedestrian = dataShaper.sort(data=pedestrian, columnsOrder=columns, ascendingFlags=ascendingFlags)
    static = dataShaper.sort(data=static, columnsOrder=columns, ascendingFlags=ascendingFlags)
    train = dataShaper.sort(data=train, columnsOrder=columns, ascendingFlags=ascendingFlags)

    for sl in sequenceLength:
        # Retrieve time series from data
        timeSeriesBus = dataShaper.retrieveSeries(data=bus, length=sl)
        timeSeriesCar = dataShaper.retrieveSeries(data=car, length=sl)
        timeSeriesPedestrian = dataShaper.retrieveSeries(data=pedestrian, length=sl)
        timeSeriesStatic = dataShaper.retrieveSeries(data=static, length=sl)
        timeSeriesTrain = dataShaper.retrieveSeries(data=train, length=sl)

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

        for n_l in r_nn_layers:
            print('FEATURES: ', i + 1, '/', 'SEQUENCE LENGTH: ', sl, '/', 'NUMBER OF LAYERS: ', n_l + 2)

            # Define model architecture
            model = keras.Sequential()
            model.add(SimpleRNN(2 * len(f), return_sequences=True, input_shape=[None, len(f)]))

            for j in range(0, n_l):
                model.add(SimpleRNN(2 * len(f), return_sequences=True))

            model.add(SimpleRNN(2 * len(f)))
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

            # Calculate predictions
            y_pred = model.predict(X_test)

            plt.plot(history.history['acc'])
            plt.plot(history.history['val_acc'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['acc', 'val_acc'], loc='upper left')
            plt.show()

            # Convert predictions and labels
            val_pred = np.argmax(y_pred, axis=1)
            val_test = np.argmax(y_test, axis=1)

            # Print classification report
            print(classification_report(val_test, val_pred))

            # Create confusion matrix
            confusion_mat = confusion_matrix(val_test, val_pred)

            # Get label names
            label_names = enc.categories_[0]
            # Visualize confusion matrix with label names
            plt.figure(figsize=(8, 6))
            sns.heatmap(confusion_mat, annot=True, fmt="d", xticklabels=label_names, yticklabels=label_names)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.show()

print('breakpoint')