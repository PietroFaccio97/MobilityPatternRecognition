import pandas as pd
import glob

class DataReader:

    def __init__(self, columns, patterns):
        self.columns = columns
        self.patterns = patterns

    def hasPatternData(self, path):
        return path in self.patterns

    def retrievePatternData(self, path):
        if path in self.patterns:
            files = glob.glob(path.value)
            boh = self.fetchCsv(files)
            return pd.concat(boh, axis=0, ignore_index=True)
        else:
            return False

    def retrieveAllData(self):
        data = pd.DataFrame
        for path in self.patterns:
            data = data.append(self.retrievePatternData(path))
        return data

    def fetchCsv(self, files):
        elements = []
        for f in files:
            elements.append(pd.read_csv(f, usecols=self.columns, sep=';'))
        return elements