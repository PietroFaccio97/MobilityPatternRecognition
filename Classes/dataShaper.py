import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def scale(data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(data)

def sort(data, columnsOrder, ascendingFlags):
    return data.sort_values(columnsOrder, ascending=ascendingFlags, inplace=False)

def roundTimestamp(data, order):
    data['timestamp'] = data['timestamp'].apply(lambda x: round(float(str(x)[:13]) / order) * order)
    return data

def retrieveContinuousSeries(data, length, deltaTime):
    timeSeries = []
    IDs = np.unique(data['ue_ident'])
    for ID in IDs:
        measures = pd.DataFrame(data.loc[data['ue_ident'] == ID])
        measures = roundTimestamp(measures, deltaTime)

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