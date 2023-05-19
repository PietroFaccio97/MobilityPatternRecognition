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
            # Check if the series contains time continuous measures
            next_valid_index = checkContinuity(measures.iloc[index:index+length])
            if next_valid_index == 0:
                # Series is continuous, it is added
                timeSeries.append(measures.iloc[index:index + length])
                index += 1
            else:
                # There is a missing measure, we move after it
                index += next_valid_index
    return timeSeries

def checkContinuity(measures):
    i = 1
    while i < len(measures):
        previous = measures.iloc[i-1:i]
        follow = measures.iloc[i:i+1]
        if follow['timestamp'].values[0] != previous['timestamp'].values[0] + 100:
            return i
        i += 1
    return 0

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
def labelAndShapeSeries(data, columns, label):
    columns = columns[2:]
    X = []
    y = []

    for series in data:
        # Extracts only the features
        X.append(series[columns])
        # Match the series to their label
        y.append(label)

    return X, y