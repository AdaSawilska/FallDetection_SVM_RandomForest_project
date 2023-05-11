import numpy as np
import pandas as pd
import glob

# reads the csv files to dataFrame
def readCSVdata(path, id_col):
    data = pd.read_csv(path, index_col=id_col)
    return data


# reads multiple csv files to one dataFrame
def readMultipleCSVdata(path):
    data_files = glob.glob(f'{path}/*.csv')
    # data = []
    # for filename in data_files:
    #     df = pd.read_csv(filename, index=filename, header=None)
    #     data.append(df)

    #data = pd.concat((pd.read_csv(filename).assign(source=filename) for filename in data_files), ignore_index=True)
    for filename in data_files:
        data = pd.read_csv(filename).assign(source=filename)
        #data.set_index([pd.Index(data.index.values), 'source'])
    return data


class_data = readCSVdata('./IFMBE Scientific Challenge/files.csv', id_col=0)
param_data = readMultipleCSVdata('./IFMBE Scientific Challenge/Train2')
print('done')
