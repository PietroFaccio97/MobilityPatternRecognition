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
        while index < len(measures):
        # for index in range(0, len(measures)):
            skip = 0
            sample = []
            if index + length < len(measures):
                for i in range(0, length):
                    previous = measures.iloc[index+i:index+i+1]
                    follow = measures.iloc[index+i+1:index+i+2]
                    if follow['timestamp'].values[0] == previous['timestamp'].values[0] + deltaTime:
                        sample.append(follow)
                    else:
                        skip = i
                        break
            if len(sample) == length:
                timeSeries.append(sample)
                index += (length - 1)
            else:
                if skip != 0:
                    index += skip
                else:
                    index += 1

    return timeSeries