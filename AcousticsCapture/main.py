# import matplotlib.pyplot as plt
import numpy as np
import csv
from scipy import signal
import math
import saleae
import time
import os

def read_file_mat(filename, numHeaders=2):
    reader = csv.reader(open(filename), delimiter=',')
    total = [[], [], [], []]
    for row in reader:
        try:
            for i in range(len(row)):
                total[i].append([float(row[i])])
        except:
            pass
    channels = [col for col in total[1:len(total)]]
    return channels


def movingAverage(values, window):
    weights = np.repeat(1.0, window) / window
    sma = np.convolve(values, weights, 'valid')
    return sma


def moving_average(arr, n):
    return np.append(np.array([np.mean(arr[:i + 1]) for i in range(n - 1)]),
                     np.correlate(arr, np.ones(n, dtype=float) / n))


def detect_wavefront(signal):
    # tune these two constants to make it more robust / reduce lag (increasing sn will reduce false positives, but increase lag)
    n = 10000
    sn = 1000
    thresh = 3  # tune this to similarly adjust robustness

    amplitude = np.abs(signal)

    longma = moving_average(amplitude, n)
    shortma = moving_average(amplitude, sn)
    return np.argmax(shortma > thresh * longma)


def findBearing(capturePath, pingerFrequency, fs, sampleTime):
    # Matrix = read_file_mat('dat1-left45.csv')
    Matrix = read_file_mat(capturePath)

    [b, a] = signal.cheby2(3, 3, [((pingerFrequency - 8) / fs * 2), ((pingerFrequency + 8) / fs * 2)], btype='bandpass')
    fil_ch1 = signal.lfilter(b, a, np.asarray(Matrix[0]), axis=0)
    # plt.plot(fil_ch1)
    # plt.show()

    start = detect_wavefront(fil_ch1[:, 0])
    print("detected wavefront in channel 0 at index %d, %.2fs" % (start, (start/len(fil_ch1))*sampleTime))

    ra = 2556 # range constant
    Channel0 = np.array(np.asarray(Matrix[0]))[start - 1:start + ra - 1]  # 241925:242005  113920:114000
    Channel1 = np.array(np.asarray(Matrix[1]))[start - 1:start + ra - 1]
    Channel2 = np.array(np.asarray(Matrix[2]))[start - 1:start + ra - 1]

    FiltChannel0 = signal.lfilter(b, a, Channel0)
    FiltChannel1 = signal.lfilter(b, a, Channel1)
    FiltChannel2 = signal.lfilter(b, a, Channel2)

    diff11 = corr(FiltChannel1, FiltChannel0, fs,
                  pingerFrequency)  ##if all bearings are opposite of expected, switch order of 2 and 1
    diff12 = corr(FiltChannel2, FiltChannel0, fs,
                  pingerFrequency)  ##if all bearings are opposite of expected, switch order of 3 and 1

    bearing = np.arctan2(diff11, diff12) * 180 / np.pi
    # print("bearing is: %.5f" % bearing)
    return bearing


def corr(waveA, waveB, fs, pingerFrequency):
    maxLag = fs / pingerFrequency
    n = len(waveA) - math.floor(maxLag)
    corrArray = []
    # print(waveA)
    # print(waveB)
    for i in range(math.floor(maxLag - 1)):  # DO YOU NEED TO SUBTRACT ONE?
        # print(np.corrcoef(waveA[:n,0], waveB[i:n+i,0]))
        corrArray.append(np.corrcoef(waveA[:n, 0], waveB[i:n + i, 0])[0][1])
    maxCorr = corrArray.index(max(corrArray))  ##MAY NEED TO SUBTRACT ONE??
    # print(corrArray)

    if maxCorr > 7.09:  ##This should theoretically be 7.09 to 7.1, but experimentally (Due to speed of sound in different water ) is changed to 7.5.
        maxCorr = maxCorr - maxLag
        # if maxCorr<-7.09: ##this should also be -7.09
    # raise("error: not possible")   may not need it
    # print(maxCorr)
    return maxCorr


def getPingerData(host='localhost', port=10429):
    s = saleae.Saleae()
    digital = []
    analog = [0, 1, 2]
    s.set_active_channels(digital, analog)
    s.set_sample_rate((0, 625000))
    s.set_capture_seconds(0.3)
    s.capture_start_and_wait_until_finished()

    file_path_on_target_machine = str("/home/robot/pingCaptures/" + str(time.time()) + ".csv")
    s.export_data2(file_path_on_target_machine,
                   digital_channels=digital,
                   analog_channels=analog,
                   format="csv",
                   analog_format="voltage"
                   )
    return file_path_on_target_machine

def startLogic():
    os.system("Logic &")
    time.sleep(12)

def mainFunction():
    # TimeOfPingOffset = recordFirstListen()
    # getPingerData(TimerClock=TimeOfPingOffset)
    # findPhaseDifference(pingerFrequency, 625000)
    # print('starting')
    # print('found bearing: %d' % findBearing(35000, 625000, 2.2))
    startLogic()
    print('begin live capture and detect')
    print('found bearing: %d' % findBearing(getPingerData(), 35000, 625000, 2.2))
    # print('done')


mainFunction()
