import numpy as np
import math
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import copy


def spectral_energy(signal, sampling_rate):

    # Convert the Series to a numpy array
    data = signal.values
    # Compute the FFT
    fft_result = np.fft.fft(signal)

    # Calculate the power spectrum
    power_spectrum = np.abs(fft_result) ** 2

    # Compute the spectral energy
    spectral_energy = np.sum(power_spectrum)

    # Determine the frequency corresponding to the maximum power
    max_power_index = np.argmax(power_spectrum)
    frequency = np.fft.fftfreq(len(data), 1/sampling_rate)
    principal_frequency = np.abs(frequency[max_power_index])

    return spectral_energy, principal_frequency


def calculate_inclination_angle_x(x, y, z):

    # Calculate the magnitude of the accelerometer vector
    magnitude = math.sqrt(x**2 + y**2 + z**2)

    # Calculate the inclination angle using trigonometry
    if magnitude == 0:
        return 0  # Handle division by zero

    inclination_rad = math.acos(x / magnitude)

    return inclination_rad


def calculate_inclination_angle_y(x, y, z):

    # Calculate the magnitude of the accelerometer vector
    magnitude = math.sqrt(x**2 + y**2 + z**2)

    # Calculate the inclination angle using trigonometry
    if magnitude == 0:
        return 0  # Handle division by zero

    inclination_rad = math.acos(y / magnitude)

    return inclination_rad


def calculate_inclination_angle_z(x, y, z):

    # Calculate the magnitude of the accelerometer vector
    magnitude = math.sqrt(x**2 + y**2 + z**2)

    # Calculate the inclination angle using trigonometry
    if magnitude == 0:
        return 0  # Handle division by zero

    inclination_rad = math.acos(z / magnitude)

    return inclination_rad


def calculate_dtw_distance(series1, series2):

    # Convert the Series to numpy arrays
    array1 = series1.values.reshape(-1, 1)
    array2 = series2.values.reshape(-1, 1)

    # Calculate the DTW distance
    distance, _ = fastdtw(array1, array2, dist=euclidean)

    return distance


def mean_standart_dev_mobile(signal, time, w, d):
    # -----
    # n : number of samples signal
    # w : lengh of window
    # d : distance between windows
    # -----
    n = len(signal)
    nb_fenetres = int((n-w)/d) + 1
    print("nd: ", nb_fenetres)
    time_moy, signal_moy = [None for _ in range(nb_fenetres)], [
        [None, None] for _ in range(nb_fenetres)]
    for i in range(len(signal_moy)):
        index = i*d+int(d/2)
        borne_inf, borne_max = int(max(0, index-w/2))+1, int(min(n, index+w/2))
        # print(signal_moy[i])
        # print(signal_moy[i][0])
        signal_moy[i][0] = sum(signal[borne_inf:borne_max+1])/w
        var = 0
        for j in range(borne_inf, borne_max+1):
            var += (signal[j]-signal_moy[i][0])**2
        var = (var/w)**0.5
        signal_moy[i][1] = var
        time_moy[i] = time[index]
    return time_moy, signal_moy


def difference_mean(signal, time, w, d):
    # -----
    # n : nombre échantillons signal
    # w : largur de la fenêtre
    # d : décalgage de la fenêtre
    # -----
    n = len(signal)
    nb_fenetres = int((n-w)/d) + 1
    #print("nd: ", nb_fenetres)
    time_moy, signal_moy, signal_diff_moy = [None for _ in range(nb_fenetres)], [
        None for _ in range(nb_fenetres)], [None for _ in range(nb_fenetres)]
    for i in range(len(signal_moy)):
        index = i*d+int(d/2)
        borne_inf, borne_max = int(max(0, index-w/2))+1, int(min(n, index+w/2))
        # print(signal_moy[i])
        # print(signal_moy[i][0])
        signal_moy[i] = np.polyfit(
            time[borne_inf:borne_max+1], signal[borne_inf:borne_max+1], 0)
        time_moy[i] = time[index]
    for i in range(len(signal_moy)):
        signal_diff_moy[i] = abs(
            signal_moy[min(i+1, len(signal_moy)-1)]-signal_moy[max(0, i-1)])

    # la sortie est un vecteur à un élément
    return time_moy, signal_diff_moy


def detection_of_elbow(signal, M):
    # M = nb points - size d'une fenêtre
    # plt.plot(signal)
    # plt.show()
    # determiner seuil
    signal_sorted = copy.deepcopy(signal)
    signal_sorted.sort()
    var = (np.var(signal_sorted[:int(0.9*len(signal_sorted))]))**0.5
    mean = np.mean(signal_sorted[:int(0.9*len(signal_sorted))])
    threshold = mean + var*math.sqrt(2*math.log(M))

    #detect + decide
    decision = np.zeros(len(signal))
    for i in range(len(signal)):
        # print(signal[i])
        if signal[i] >= threshold:
            # aover the threshold
            if (signal[i] > signal[i-1] and signal[i] >= signal[i+1]) or (signal[i] >= signal[i-1] and signal[i] > signal[i+1]):
                # max local
                decision[i] = 1


# 1 -> smooth the signal using mean_standart_dev_mobile function
# 2 -> apply the difference_mean function to get the signal we will analyse to find changes in slope
# 3 -> use detection_of_elbow to detect the changes of slope in the signal
# 4 -> the decision_of_elbos function returns a list containing 0s where there was no change of slope and 1s where there were.
# 5 -> count the ones andyou get the number of slope changes
