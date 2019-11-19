import pandas
# correlate(in1, in2) = k: in2 faster than in1 by k
from scipy.signal import correlate, cheby2, lfilter, freqz
import scipy.signal as signal
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import numpy as np
import math
import sys
import fft

#sampling frequency
fs = 625000
#sampling time period
ts = 3
#number of sample taken during the ping
pingc = fs*0.004
#target frequency
freq = 40000
#speed of sound in water
vsound = 1511.5
#nipple distance between hydrophone
spac = 0.0115
#allowed phase diff
dphase = math.pi*2/(vsound/freq)*spac

fft_w_size = 125

bw = 1600*3


def data_to_mag(data):
    plist, mlist = fft.fft(data, freq, fft_w_size)
    maxi = fft.moving_average_max(mlist)
    return np.sum(mlist[maxi : maxi+int(pingc/fft_w_size)])


if __name__ == "__main__":

    path = "/Users/estellehe/Documents/senior/IndepStudy/wilson_range/625k_40k_.csv"
    range_coord = [[240, 30, 198], [140, 30, 192], [80, 30, 186]] # inch

    # df = pandas.read_csv(filepath, skiprows=[1], skipinitialspace=True)
    # # print("running ", filepath)
    # data1 = df["Channel 0"].tolist()
    # data2 = df["Channel 1"].tolist()
    # data3 = df["Channel 2"].tolist()
    # data4 = df["Channel 3"].tolist()
    mag1a = []
    mag2a = []
    mag3a = []
    mag4a = []
    dista = []
    for r in range_coord:
        filepath = path.replace(".csv", str(r[0])+"_"+str(r[1])+"_"+str(r[2])+".csv")
        dist = np.sqrt(r[0]^2+r[1]^2+r[2]^2)
        mag1 = []
        mag2 = []
        mag3 = []
        mag4 = []
        for i in range(4):
            data1, data2, data3, data4 = fft.read_data(filepath.replace(".csv", "("+str(i+1)+").csv"))
            mag11 = data_to_mag(data1[0:int(len(data1)/2)])
            mag21 = data_to_mag(data2[0:int(len(data1)/2)])
            mag31 = data_to_mag(data3[0:int(len(data1)/2)])
            mag41 = data_to_mag(data4[0:int(len(data1)/2)])
            mag12 = data_to_mag(data1[int(len(data1)/2):len(data1)])
            mag22 = data_to_mag(data2[int(len(data1)/2):len(data1)])
            mag32 = data_to_mag(data3[int(len(data1)/2):len(data1)])
            mag42 = data_to_mag(data4[int(len(data1)/2):len(data1)])

            mag1.extend([mag11, mag12])
            mag2.extend([mag21, mag22])
            mag3.extend([mag31, mag32])
            mag4.extend([mag41, mag42])
            dista.extend([dist, dist])
        mag1a.extend(mag1)
        mag2a.extend(mag2)
        mag3a.extend(mag3)
        mag4a.extend(mag4)
        print("mag1", np.mean(mag1), "mag2", np.mean(mag2), "mag3", np.mean(mag3), "mag4", np.mean(mag4))

        plt.figure()
        plt.plot(range(len(mag1)), mag1)
        plt.plot(range(len(mag1)),mag2)
        plt.plot(range(len(mag1)),mag3)
        plt.plot(range(len(mag1)),mag4)
        plt.legend(['1', '2', '3', '4'])
        plt.title("range: "+str(r))



    plt.figure()
    plt.scatter(dista, mag1a)
    plt.title("Channel 1")
    plt.figure()
    plt.scatter(dista, mag2a)
    plt.title("Channel 2")
    plt.figure()
    plt.scatter(dista, mag3a)
    plt.title("Channel 3")
    plt.figure()
    plt.scatter(dista, mag4a)
    plt.title("Channel 4")
    # plt.legend(['1', '2', '3', '4'])
    plt.show()

    print("mag1", np.mean(mag1a), "mag2", np.mean(mag2a), "mag3", np.mean(mag3a), "mag4", np.mean(mag4a))
