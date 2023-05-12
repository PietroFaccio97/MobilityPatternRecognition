import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

#TODO -> convert to class with empty constructor and static methods

def scale(data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(data)

def sort(data, columnsOrder, ascendingFlags):
    return data.sort_values(columnsOrder, ascending=ascendingFlags, inplace=False)

def retrieveSeries(data, length):
    timeSeries = []
    # Distinguish samples based on device
    IDs = np.unique(data['ue_ident'])

    for ID in IDs:
        # Retrieving measures for the device
        measures = pd.DataFrame(data.loc[data['ue_ident'] == ID])

        index = 0
        # Prevent index out of bounds
        while index + length <= len(measures):
            # Append a sequential number of measures based on the input length
            if checkContinuity(measures.iloc[index:index+length]):
                timeSeries.append(measures.iloc[index:index+length])
            # Move to the next series (it is a sliding windows mechanism so samples are repeated)
            index += 1

    return timeSeries

def checkContinuity(measures):
    i = 0
    while i < len(measures) - 1:
        previous = measures.iloc[i:i+1]
        follow = measures.iloc[i+1:i+2]
        if follow['timestamp'].values[0] != previous['timestamp'].values[0] + 100:
            return False
        i += 1
    return True

def retrieveSlidingSeries(data, length, deltaTime):
    timeSeries = []
    IDs = np.unique(data['ue_ident'])
    for ID in IDs:
        measures = pd.DataFrame(data.loc[data['ue_ident'] == ID])
        #measures = roundTimestamp(measures, deltaTime)

        index = 0
        while index + length <= len(measures):
            sample = []

            start = measures.iloc[index:index+1]
            sample.append(start)

            skip = 1
            i = index
            while i <= len(measures):
                previous = measures.iloc[i:i+1]
                follow = measures.iloc[i+1:i+2]
                if follow['timestamp'].values[0] == previous['timestamp'].values[0] + deltaTime:
                    sample.append(follow)
                else:
                    index += skip
                    break
                if len(sample) == length:
                    timeSeries.append(sample)
                    index += length
                    break
                skip += 1
                i += 1

    return timeSeries

def retrieveContinuousSeries(data, length, deltaTime):
    timeSeries = []
    IDs = np.unique(data['ue_ident'])
    for ID in IDs:
        measures = pd.DataFrame(data.loc[data['ue_ident'] == ID])
        #measures = roundTimestamp(measures, deltaTime)

        index = 0
        while index + length <= len(measures):
            sample = []

            start = measures.iloc[index:index+1]
            sample.append(start)

            skip = 1
            i = index
            while i <= len(measures):
                previous = measures.iloc[i:i+1]
                follow = measures.iloc[i+1:i+2]
                if follow['timestamp'].values[0] == previous['timestamp'].values[0] + deltaTime:
                    sample.append(follow)
                else:
                    index += skip
                    break
                if len(sample) == length:
                    timeSeries.append(sample)
                    index += length
                    break
                skip += 1
                i += 1

    return timeSeries

# Prepare the measure series for the dataset
def labelAndShapeSeries(data, label):
    X = []
    y = []

    for series in data:
        # Extracts only the features
        X.append(series['phy_ul_pucch_rssi'])
        # Match the series to their label
        y.append(label)

    return X, y