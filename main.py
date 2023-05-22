import os
import scipy
from pandas.plotting import scatter_matrix
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.signal import spectrogram
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score


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


# find feature for all signal together
def find_global_features(data_files):
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


def find_local_features(df, filename, signals_dataframe):
    description_i = df.describe()
    signals_dataframe['mean_x'][filename] = description_i['x']['mean']
    signals_dataframe['std_x'][filename] = description_i['x']['std']
    signals_dataframe['min_x'][filename] = description_i['x']['min']
    signals_dataframe['max_x'][filename] = description_i['x']['max']
    signals_dataframe['mean_y'][filename] = description_i['y']['mean']
    signals_dataframe['std_y'][filename] = description_i['y']['std']
    signals_dataframe['min_y'][filename] = description_i['y']['min']
    signals_dataframe['max_y'][filename] = description_i['y']['max']
    signals_dataframe['mean_z'][filename] = description_i['z']['mean']
    signals_dataframe['std_z'][filename] = description_i['z']['std']
    signals_dataframe['min_z'][filename] = description_i['z']['min']
    signals_dataframe['max_z'][filename] = description_i['z']['max']

    return signals_dataframe


def normalize(data, global_features_x, global_features_y, global_features_z):
    mean_x = float(global_features_x['mean'])
    std_x = float(global_features_x['std'])
    mean_y = float(global_features_y['mean'])
    std_y = float(global_features_y['std'])
    mean_z = float(global_features_z['mean'])
    std_z = float(global_features_z['std'])

    normalised_signal = pd.DataFrame(columns=["x", "y", "z", "time"])
    normalised_signal["x"] = (data["x"] - mean_x) / std_x
    normalised_signal["y"] = (data["y"] - mean_y) / std_y
    normalised_signal["z"] = (data["z"] - mean_z) / std_z
    normalised_signal['time'] = data['time']
    return normalised_signal


# filter signal
def filter(order, fs, cutoff):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def signal_filtration(data, order, fs, cutoff):
    b, a = filter(order, fs, cutoff)
    filtered_signal = pd.DataFrame(columns=["x", "y", "z", "time"])
    filtered_signal["x"] = signal.filtfilt(b, a, data['x'])
    filtered_signal["y"] = signal.filtfilt(b, a, data['y'])
    filtered_signal["z"] = signal.filtfilt(b, a, data['z'])
    return filtered_signal


def signal_spectometer(data, timeStep):
    yf = fft(data)
    xf = fftfreq(len(data), d=timeStep)

    plt.plot(xf, np.abs(yf))
    plt.xlabel('frequency Hz')
    plt.ylabel('power ')
    plt.show()


# reads multiple csv files to one dataFrame
def preprocessing(data_files, global_features_x, global_features_y, global_features_z, labels):
    # extract the filenames from path
    data_names = []
    for d in data_files:
        headtail = os.path.split(d)
        data_names.append(headtail[1])

    # create dataframe for statistical parameters of the signal
    signals = pd.DataFrame(data=[], index=[data_names],
                           columns=["mean_x", "std_x", "min_x", "max_x", "mean_y", "std_y", "min_y", "max_y", "mean_z",
                                    "std_z", "min_z", "max_z", "class"])
    # data = []
    for filepath, filename in zip(data_files, data_names):
        df = pd.read_csv(filepath, names=['x', 'y', 'z', 'time'])

        # normalize the signal from certain file - NOT all at the same time
        normalized_df = normalize(df, global_features_x, global_features_y, global_features_z)

        # count some time related parameters
        duration = float((df['time'][[len(df['time']) - 1]] - df['time'][0]))
        timeStep = float((duration / len(df['time'])))
        fs = float(1 / timeStep)
        sampleRate = len(df) / duration

        # filter parameters
        order = 5
        cutoff = 2

        # signal filtration
        filtered_df = signal_filtration(normalized_df, order, fs, cutoff)

        # count parameters of each signals and save to dataframe
        signals = find_local_features(filtered_df, filename, signals)
        signals['class'][filename] = labels['class'][filename]


        # # plot raw signal
        # plt.plot(df['time'], df['x'], label='x')
        # plt.plot(df['time'], df['y'], label='y')
        # plt.plot(df['time'], df['z'], label='z')
        # plt.title(filename)
        # plt.xlabel('time')
        # plt.legend()
        # plt.show()
        #
        # # plot normalized signal
        # plt.plot(df['time'], normalized_df['x'], label='x-filtered')
        # plt.plot(df['time'], normalized_df['y'], label='y-filtered')
        # plt.plot(df['time'], normalized_df['z'], label='z-filtered')
        # plt.title(filename)
        # plt.xlabel('time')
        # plt.legend()
        # plt.show()
        #
        # # plot filtered signal
        # plt.plot(df['time'], filtered_df['x'], label='x-filtered')
        # plt.plot(df['time'], filtered_df['y'], label='y-filtered')
        # plt.plot(df['time'], filtered_df['z'], label='z-filtered')
        # plt.title(filename)
        # plt.xlabel('time')
        # plt.legend()
        # plt.show()
        #
        # # plot a power spectrum of the signal
        # signal_spectometer(filtered_df['x'], timeStep)

        print(f'{filename} done')
    signals = signals.astype(float)
    dataset = signals.drop('class', axis=1)
    scatter_matrix(dataset, figsize=(16, 9))
    plt.show()
    return signals


# NOT WORKING YET
def training(df):
        Y = df['class']
        X = df.drop('class', axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=25)
        svc = SVC(kernel='linear')
        svc.fit(X_train, y_train)
        y_pred = svc.predict(X_test)

        # Evaluating the accuracy of the model using the sklearn functions
        accuracy = accuracy_score(y_test.values, y_pred) * 100
        confusion_mat = confusion_matrix(y_test.values, y_pred)

        # Printing the results
        print("Accuracy for SVM is:", accuracy)
        print("Confusion Matrix")
        print(confusion_mat)
        print('svm')




if __name__ == '__main__':
    class_data = readCSVdata('./IFMBE Scientific Challenge/file_3classes.csv', id_col=1)

    # url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    # names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    # dataset = pd.read_csv(url, names=names)
    # dataset = dataset.drop('class', axis=1)
    # scatter_matrix(dataset, figsize=(16, 9))
    # plt.show()


    # changed_class_data = classChange(class_data)

    path = './IFMBE Scientific Challenge/Train2'
    data_files = glob.glob(f'{path}/*.csv')
    global_features_x, global_features_y, global_features_z = find_global_features(data_files)
    signals_parameters_df = preprocessing(data_files, global_features_x, global_features_y, global_features_z, class_data)
    training(signals_parameters_df)
    print('done')
