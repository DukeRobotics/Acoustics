# import matplotlib.pyplot as plt
import numpy as np
import csv
from scipy import signal
import math
import saleae
import time
import os
import psutil
from xvfbwrapper import Xvfb


#This should theoretically be 7.09 to 7.1, but experimentally (Due to speed of sound in different water ) is changed to 7.5.
MAX_CORR_CONST = 7.09
RANGE_CONST = 2556
PING_CAP_DIR = "/home/robot/pingCaptures/"


def readFileMat(filename, numHeaders=2):
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

def movingAverage(arr, n):
    return np.append(np.array([np.mean(arr[:i + 1]) for i in range(n - 1)]),
                     np.correlate(arr, np.ones(n, dtype=float) / n))


def detectWavefront(signal):
    # tune these two constants to make it more robust / reduce lag (increasing sn will
    # reduce false positives, but increase lag)
    n = 10000
    sn = 1000
    thresh = 3  # tune this to similarly adjust robustness

    amplitude = np.abs(signal)

    longma = movingAverage(amplitude, n)
    shortma = movingAverage(amplitude, sn)
    return np.argmax(shortma > thresh * longma)


def findBearing(capturePath, pingerFrequency, fs, sampleTime):
    # matrix = read_file_mat('dat1-left45.csv')
    print(capturePath)
    matrix = readFileMat(capturePath)

    [b, a] = signal.cheby2(3, 3, [((pingerFrequency - 8) / fs * 2), ((pingerFrequency + 8) / fs * 2)], btype='bandpass')
    fil_ch1 = signal.lfilter(b, a, np.asarray(matrix[0]), axis=0)
    # plt.plot(fil_ch1)
    # plt.show()

    start = detectWavefront(fil_ch1[:, 0])
    print("detected wavefront in channel 0 at index %d, %.2fs" % (start, (start / len(fil_ch1)) * sampleTime))

    ra = RANGE_CONST  # range constant
    channel0 = np.array(np.asarray(matrix[0]))[start - 1:start + ra - 1]  # 241925:242005  113920:114000
    channel1 = np.array(np.asarray(matrix[1]))[start - 1:start + ra - 1]
    channel2 = np.array(np.asarray(matrix[2]))[start - 1:start + ra - 1]

    filtChannel0 = signal.lfilter(b, a, channel0)
    filtChannel1 = signal.lfilter(b, a, channel1)
    filtChannel2 = signal.lfilter(b, a, channel2)

    diff11 = corr(filtChannel1, filtChannel0, fs,
                  pingerFrequency)  ##if all bearings are opposite of expected, switch order of 2 and 1
    diff12 = corr(filtChannel2, filtChannel0, fs,
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

    if maxCorr > MAX_CORR_CONST:
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
    s.set_capture_seconds(2.2)
    s.capture_start_and_wait_until_finished()

    t = time.time()
    file_path_on_target_machine = str(PING_CAP_DIR + str(t) + ".csv")
    s.export_data2(file_path_on_target_machine,
                   digital_channels=digital,
                   analog_channels=analog,
                   format="csv",
                   analog_format="voltage"
                   )
    return t, file_path_on_target_machine


def checkLogicRunning():
    for pid in psutil.pids():
        p = psutil.Process(pid)
        if "Logic" in p.cmdline() or "Logic" in p.name():
            return True
    return False
        

def startLogic():
    # Don't start if already running
    if not checkLogicRunning():
        os.system("Logic &")
        time.sleep(20)


def mainFunction():
    display = Xvfb()
    display.start()
    while True:
        startLogic()
        time, pdat = getPingerData()
        print('=====================================================')
        print('found bearing from time %d: %d degrees' % (time, findBearing(pdat, 35000, 625000, 2.2)))


mainFunction()
