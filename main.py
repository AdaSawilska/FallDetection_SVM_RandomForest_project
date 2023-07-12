import os
import scipy
from numpy import arange
from pandas.plotting import scatter_matrix
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.signal import spectrogram
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
from sklearn import svm, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, RocCurveDisplay
from added_features import *
import seaborn as sns


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


def find_local_features(df, filename, signals_dataframe, sampl_rate):
    description_i = df.describe()
    signals_dataframe['mean_x'][filename] = description_i['x']['mean']
    signals_dataframe['std_x'][filename] = description_i['x']['std']
    signals_dataframe['min_x'][filename] = description_i['x']['min']
    signals_dataframe['25%_x'][filename] = description_i['x']['25%']
    signals_dataframe['50%_x'][filename] = description_i['x']['50%']
    signals_dataframe['75%_x'][filename] = description_i['x']['75%']
    signals_dataframe['max_x'][filename] = description_i['x']['max']
    spectral_energy_x, principal_frequency_x = spectral_energy(df['x'], sampl_rate)
    signals_dataframe['spec_ener_x'][filename] = spectral_energy_x
    signals_dataframe['princ_freq_x'][filename] = principal_frequency_x
    # signals_dataframe['inclination_angle_x'][filename] = calculate_inclination_angle_x(df['x'], df['y'], df['z'])

    signals_dataframe['mean_y'][filename] = description_i['y']['mean']
    signals_dataframe['std_y'][filename] = description_i['y']['std']
    signals_dataframe['min_y'][filename] = description_i['y']['min']
    signals_dataframe['25%_y'][filename] = description_i['y']['25%']
    signals_dataframe['50%_y'][filename] = description_i['y']['50%']
    signals_dataframe['75%_y'][filename] = description_i['y']['75%']
    signals_dataframe['max_y'][filename] = description_i['y']['max']
    spectral_energy_y, principal_frequency_y = spectral_energy(df['y'], sampl_rate)
    signals_dataframe['spec_ener_y'][filename] = spectral_energy_y
    signals_dataframe['princ_freq_y'][filename] = principal_frequency_y
    # signals_dataframe['inclination_angle_y'][filename] = calculate_inclination_angle_y(df['x'], df['y'], df['z'])

    signals_dataframe['mean_z'][filename] = description_i['z']['mean']
    signals_dataframe['std_z'][filename] = description_i['z']['std']
    signals_dataframe['min_z'][filename] = description_i['z']['min']
    signals_dataframe['25%_z'][filename] = description_i['z']['25%']
    signals_dataframe['50%_z'][filename] = description_i['z']['50%']
    signals_dataframe['75%_z'][filename] = description_i['z']['75%']
    signals_dataframe['max_z'][filename] = description_i['z']['max']
    spectral_energy_z, principal_frequency_z = spectral_energy(df['z'], sampl_rate)
    signals_dataframe['spec_ener_z'][filename] = spectral_energy_z
    signals_dataframe['princ_freq_z'][filename] = principal_frequency_z
    # signals_dataframe['inclination_angle_z'][filename] = calculate_inclination_angle_z(df['x'], df['y'], df['z'])

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
def filter(order, fs, cutoff, type):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype=type, analog=False)
    return b, a


def signal_filtration(data, order, fs, cutoff, type):
    b, a = filter(order, fs, cutoff, type)
    filtered_signal = pd.DataFrame(columns=["x", "y", "z", "time"])
    filtered_signal["x"] = signal.filtfilt(b, a, data['x'])
    filtered_signal["y"] = signal.filtfilt(b, a, data['y'])
    filtered_signal["z"] = signal.filtfilt(b, a, data['z'])
    return filtered_signal


def signal_spectometer(data, timeStep):
    yf = fft(data.values)
    xf = fftfreq(len(data.values), d=timeStep)

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
                           columns=["mean_x", "std_x", "min_x", "max_x", "25%_x", "50%_x", "75%_x", 'spec_ener_x',
                                    'princ_freq_x',
                                    "mean_y", "std_y", "min_y", "max_y", "25%_y", "50%_y", "75%_y", 'spec_ener_y',
                                    'princ_freq_y',
                                    "mean_z", "std_z", "min_z", "max_z", "25%_z", "50%_z", "75%_z", 'spec_ener_z',
                                    'princ_freq_z',
                                    "class"])
    for filepath, filename in zip(data_files, data_names):
        df = pd.read_csv(filepath, names=['x', 'y', 'z', 'time'])
        class_label = labels['class'][filename]
        if class_label == 1:
            class_label = 'moving'
        elif class_label == 2:
            class_label = 'falling'
        else:
            class_label = 'other'

        # normalize the signal from certain file - NOT all at the same time
        normalized_df = normalize(df, global_features_x, global_features_y, global_features_z)

        # count some time related parameters
        duration = float((df['time'][[len(df['time']) - 1]] - df['time'][0]))
        timeStep = float((duration / len(df['time'])))
        fs = float(1 / timeStep)
        sampleRate = len(df) / duration

        # filter parameters
        order = 5
        cutoff_high = 0.15
        cutoff_low = fs / 2 - 0.3

        # plot a power spectrum of the signal
        # signal_spectometer(normalized_df['x'], timeStep)

        # signal filtration
        filtered_df = signal_filtration(normalized_df, order, fs, cutoff_low, 'low')
        filtered_df = signal_filtration(filtered_df, order, fs, cutoff_high, 'high')

        # count parameters of each signals and save to dataframe
        signals = find_local_features(filtered_df, filename, signals, sampleRate)
        signals['class'][filename] = labels['class'][filename]

        # plot raw signal
        plt.plot(df['time'], df['x'], label='x')
        plt.plot(df['time'], df['y'], label='y')
        plt.plot(df['time'], df['z'], label='z')
        plt.title(f'Label: {class_label}')
        plt.xlabel('time')
        plt.legend()
        plt.show()
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
        # plot a power spectrum of the signal
        # signal_spectometer(filtered_df['x'], timeStep)

        # print(f'{filename} done')
    signals = signals.astype(float)
    dataset = signals.drop('class', axis=1)
    corr = dataset.corr()
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corr, mask=np.zeros_like(corr),
                cmap=sns.diverging_palette(220, 10, as_cmap=True),
                square=True, ax=ax)
    plt.savefig('corr_selected_0.01.png')
    plt.show()
    # scatter_matrix(dataset, figsize=(16, 9))
    # plt.show()
    #
    # fig = plt.figure(figsize=(8, 8))
    # plt.scatter(signals['std_x'], signals['std_z'], c=signals['class'])
    # plt.show()
    return signals


# implementing SVM and random forest
def learning(df, kernel, n_estimator):
    Y = df['class']
    X = df.drop('class', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=25, stratify=Y)

    features_selected = Feature_selection(X_train, y_train, 10)
    print(features_selected)
    # accuracy_df = SVM_classifier(X_train, X_test, y_train, y_test, 'rbf')
    accuracy_df = RandomForest_classifier(X_train.loc[:, features_selected], X_test.loc[:, features_selected],
                                          y_train, y_test, n_estimator)
    return accuracy_df


def Feature_selection(X_train, y_train, n_estim):
    select = SelectFromModel(RandomForestClassifier(n_estimators=n_estim), threshold=0.01)
    select.fit(X_train, y_train)
    imp = select.estimator_.feature_importances_
    select.get_support()
    selected_feat = X_train.columns[(select.get_support())]
    forest_importances = pd.Series(imp, index=X_train.columns)

    fig, ax = plt.subplots()
    forest_importances.plot.bar(ax=ax)
    ax.set_title("Feature importances")
    ax.set_ylabel("Importance of the feature")
    fig.tight_layout()
    plt.savefig('feature_importance_selected.png')
    print('sel')

    features = X_train.columns[(select.get_support())]
    X_train_reduced = X_train.loc[:, X_train.columns[(select.get_support())]]

    return features


def RandomForest_classifier(X_train, X_test, y_train, y_test, n_estimator):
    clf = RandomForestClassifier(n_estimators=n_estimator)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test.values, y_pred) * 100

    confusion_matrix = metrics.confusion_matrix(y_test.values, y_pred)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
    cm_display.plot()
    plt.savefig('ConfusionMatrix_selected_0.01.png')
    plt.show()

    # print("Validation accuracy:", accuracy)
    # print("Confusion Matrix")
    # print(confusion_mat)
    # print("Random Forest")
    return accuracy


def SVM_classifier(X_train, X_test, y_train, y_test, kernel):
    svm = SVC(kernel=kernel)

    if kernel == 'rbf':
        # finding best params for rbf kernel
        param_grid = {'gamma': [0.005, 0.01, 0.1], 'C': [10, 15, 20, 25]}

        # Perform grid search with cross-validation
        grid_search = GridSearchCV(svm, param_grid, cv=5)
        grid_search.fit(X_train, y_train)

        # Print best parameter values and accuracy on validation set
        print("Best gamma value: ", grid_search.best_params_['gamma'])
        print("Best C value: ", grid_search.best_params_['C'])
        y_pred = grid_search.predict(X_test)
        accuracy = accuracy_score(y_test.values, y_pred) * 100
        # confusion_mat = confusion_matrix(y_test.values, y_pred)
        print("Validation accuracy: ", accuracy)
        print("Confusion Matrix")
        # print(confusion_mat)

    elif kernel == 'linear':
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)

        # Evaluating the accuracy of the model using the sklearn functions
        accuracy = accuracy_score(y_test.values, y_pred) * 100
        # confusion_mat = confusion_matrix(y_test.values, y_pred)

        # Printing the results
        print("Validation accuracy:", accuracy)
        print("Confusion Matrix")
        # print(confusion_mat)
    return accuracy
    print('svm')


if __name__ == '__main__':
    class_data = readCSVdata('./IFMBE Scientific Challenge/file_3classes.csv', id_col=1)
    # changed_class_data = classChange(class_data)
    path = './IFMBE Scientific Challenge/Train2'
    data_files = glob.glob(f'{path}/*.csv')
    global_features_x, global_features_y, global_features_z = find_global_features(data_files)
    signals_parameters_df = preprocessing(data_files, global_features_x, global_features_y, global_features_z,
                                          class_data)
    n_estimators = [22]
    accuracy = []

    for e in n_estimators:
        acc = 0
        for i in range(1):
            acc += learning(signals_parameters_df, 'rbf', e)

        mean_acc = acc / 1
        accuracy.append(mean_acc)
        print(f'Mean accuracy: {mean_acc}')
    print('done')
    # plt.figure()
    # plt.plot(n_estimators, accuracy)
    # plt.xlabel('Number of estimators')
    # plt.ylabel('Accuracy')
    # plt.show()
