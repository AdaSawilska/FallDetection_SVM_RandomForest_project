import scipy
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.signal import spectrogram
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import pylab as py


# reads the csv files to dataFrame
def readCSVdata(path, id_col):
    data = pd.read_csv(path, index_col=id_col)
    return data


#
def classChange(df):
    df.loc[df['class'] == 1, 'class'] = 1
    df.loc[df['class'] == 2, 'class'] = 1
    df.loc[df['class'] == 3, 'class'] = 1
    df.loc[df['class'] == 4, 'class'] = 3
    df.loc[df['class'] == 5, 'class'] = 3
    df.loc[df['class'] == 6, 'class'] = 2
    df.loc[df['class'] == 7, 'class'] = 2
    df.loc[df['class'] == 8, 'class'] = 2
    df.loc[df['class'] == 9, 'class'] = 3
    df.loc[df['class'] == 10, 'class'] = 3

    df.to_csv('./file_3classes.csv')

    return df


# filter signal
def filter(order, fs, cutoff):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def signal_filtration(data, order, fs, cutoff):
    b, a = filter(order, fs, cutoff)
    y = signal.filtfilt(b, a, data)
    return y


def signal_spectometer(data, timeStep ):


    yf = fft(data)
    xf = fftfreq(len(data), d=timeStep)

    plt.plot(xf, np.abs(yf))
    plt.xlabel('frequency Hz')
    plt.ylabel('power ')
    plt.show()




# reads multiple csv files to one dataFrame
def readMultipleCSVdata(path, global_features_x, global_features_y, global_features_z):
    data_files = glob.glob(f'{path}/*.csv')
    # data = []
    for filename in data_files:
        df = pd.read_csv(filename, names=['x', 'y', 'z', 'time'])

        normalized_df = normalize(df, global_features_x, global_features_y, global_features_z)







        duration = float((df['time'][[len(df['time'])-1]]-df['time'][0]))
        timeStep = float(((df['time'][[len(df['time'])-1]]-df['time'][0])/len(df['time'])))
        fs = float(1/timeStep)
        sampleRate = len(df)/duration
        order = 5
        cutoff = 2
        x_filtered = signal_filtration(df['x'], order, fs, cutoff)
        y_filtered = signal_filtration(df['y'], order, fs, cutoff)
        z_filtered = signal_filtration(df['z'], order, fs, cutoff)
        plt.plot(df['time'], df['x'], label='x')
        plt.plot(df['time'], df['y'], label='y')
        plt.plot(df['time'], df['z'], label='z')
        plt.title(filename)
        plt.xlabel('time')
        plt.legend()
        plt.show()

        plt.plot(df['time'], normalized_df['x'], label='x-filtered')
        plt.plot(df['time'], normalized_df['y'], label='y-filtered')
        plt.plot(df['time'], normalized_df['z'], label='z-filtered')
        plt.title(filename)
        plt.xlabel('time')
        plt.legend()
        plt.show()

        # plt.plot(df['time'], x_filtered, label='x-filtered')
        # plt.plot(df['time'], y_filtered, label='y-filtered')
        # plt.plot(df['time'], z_filtered, label='z-filtered')
        # plt.title(filename)
        # plt.xlabel('time')
        # plt.legend()
        # plt.show()
        signal_spectometer(x_filtered, timeStep)
        print('ok')


# find feature for all signal together
def find_features(path):
    data_files = glob.glob(f'{path}/*.csv')
    global_description_x = pd.DataFrame(data=[], index=[
                               "count", "mean", "std", "min", "25%", "50%", "75%", "max"], columns=[])
    global_description_y = pd.DataFrame(data=[], index=[
                               "count", "mean", "std", "min", "25%", "50%", "75%", "max"], columns=[])
    global_description_z = pd.DataFrame(data=[], index=[
                               "count", "mean", "std", "min", "25%", "50%", "75%", "max"], columns=[])
    for filename in data_files:
        df = pd.read_csv(filename, names=['x', 'y', 'z', 'time'])
        description_i = df.describe()
        global_description_x[filename] = description_i["x"]
        global_features_x = global_description_x.mean(axis=1)
        
        global_description_y[filename] = description_i["y"]
        global_features_y = global_description_y.mean(axis=1)

        global_description_z[filename] = description_i["z"]
        global_features_z = global_description_z.mean(axis=1)

    return global_features_x, global_features_y, global_features_z

def normalize(data, global_features_x, global_features_y, global_features_z):
   
    mean_x = float(global_features_x['mean'])
    std_x = float(global_features_x['std'])
    mean_y = float(global_features_y['mean'])
    std_y = float(global_features_y['std'])
    mean_z = float(global_features_z['mean'])
    std_z = float(global_features_z['std'])
    
    normalised_signal= pd.DataFrame(columns = ["x","y","z"])
    normalised_signal["x"]  = (data["x"] - mean_x)/std_x
     
    normalised_signal["y"]  = (data["y"] - mean_y)/std_y
  
    normalised_signal["z"]  = (data["z"] - mean_z)/std_z
    normalised_signal['time'] = data['time']
    return normalised_signal



if __name__ == '__main__':
    class_data = readCSVdata('./IFMBE Scientific Challenge/file_3classes.csv', id_col=0)

    #changed_class_data = classChange(class_data)
    path = './IFMBE Scientific Challenge/Train2'
    global_features_x, global_features_y, global_features_z = find_features(path)
    readMultipleCSVdata(path, global_features_x, global_features_y, global_features_z)


    print('done')
