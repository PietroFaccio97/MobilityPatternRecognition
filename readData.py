import pandas as pd
import glob


# Columns to read
columns = ['ue_ident', 'timestamp', 'phy_ul_pucch_rssi', 'phy_ul_pucch_rssi']

def retrieveData(path):
    files = glob.glob(path.value)
    boh = fetchCsv(files)
    return pd.concat(boh, axis=0, ignore_index=True)

def fetchCsv(files):
    elements = []
    for f in files:
        elements.append(pd.read_csv(f, usecols=columns, sep=';'))
    return elements