import pandas as pd
import glob

from Static.typeEnum import Paths


class DataReader:

    def __init__(self, columns, patterns, outliers, outlierThresholds):
        self.columns = columns
        self.patterns = patterns
        self.outliers = outliers
        self.outlierThresholds = outlierThresholds

    def hasPatternData(self, path):
        return path in self.patterns

    def retrieveAllData(self):
        data = pd.DataFrame
        for path in self.patterns:
            data = data.append(self.retrievePatternData(path))
        return data

    #patterns = [Paths.bus, Paths.car, Paths.pedestrian, Paths.static, Paths.train]

    def retrievePatternData(self, path):
        if path in self.patterns:
            data = glob.glob(path.value)
            match path:
                case Paths.bus:
                    files = self.fetchCsv(data, 'bus')
                case Paths.car:
                    files = self.fetchCsv(data, 'car')
                case Paths.pedestrian:
                    files = self.fetchCsv(data, 'pedestrian')
                case Paths.static:
                    files = self.fetchCsv(data, 'static')
                case Paths.train:
                    files = self.fetchCsv(data, 'train')
                case _:
                    files = self.fetchCsv(data)
            return pd.concat(files, axis=0, ignore_index=True)
        else:
            return False

    def fetchCsv(self, files, pattern = None):
        elements = []
        for f in files:
            # Remove outliers from the file
            if pattern is not None:
                cleanFile = self.cleanOutliers(pd.read_csv(f, usecols=self.columns, sep=';'), pattern)
            else:
                cleanFile = self.cleanOutliers(pd.read_csv(f, usecols=self.columns, sep=';'))
            # Format timestamp
            timeFormattedFile = self.formatTimestamp(cleanFile)
            # Add file data to the set of data
            elements.append(timeFormattedFile)
        return elements

    def cleanOutliers(self, data, pattern = None):
        for outlier in self.outliers:
            # Calculate the average value of a feature
            avg = data[outlier].median()
            # Remove values far from the average (tune threshold to adjust)
            if pattern is not None:
                data.drop(data[abs(abs(avg) - abs(data[outlier])) > self.outlierThresholds[pattern][outlier]].index, inplace=True)
            else:
                data.drop(data[abs(abs(avg) - abs(data[outlier])) > self.outlierThresholds[outlier]].index, inplace=True)
            # Reset index
            #data.reset_index(inplace=True)
        return data

    def formatTimestamp(self, data):
        data['timestamp'] = data['timestamp'].apply(lambda x: round(float(str(x)[:13]) / 100) * 100)
        return data