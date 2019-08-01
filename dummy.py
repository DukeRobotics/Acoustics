import os
import pandas
import threading
import time
import sys
from scipy import signal
import matplotlib.pyplot as plt
from scipy.signal import correlate, cheby2, lfilter
import numpy as np



freq = 35000

fs = 625000

order = 5

bw = 2400

k = 60

# 40000hz bw=1600 k=60 order=5
# 35000hz bw=2400 k=60 order=5
# 30000hz bw=1400 k=20 order=5
# 25000hz bw=1400 k=10 order=5



df = pandas.DataFrame()
temp_path = "/tmp/dummy.csv"

def pandas_read():
    global df
    df = pandas.read_csv(temp_path, skiprows=[1], skipinitialspace=True)
    return

#bandwidth need to be 800*2
def cheby2_bandpass(lowcut, highcut, fs, order=order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = cheby2(order, k, [low, high], btype='bandpass')
    return b, a

#filter the data with bandpass
def cheby2_bandpass_filter(data, lowcut, highcut, fs, order=order):
    b, a = cheby2_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

if __name__ == "__main__":
    # # read = threading.Thread(target=pandas_read)
    # # read.start()
    # # read.join()
    # df = pandas.read_csv(temp_path, skiprows=[1], skipinitialspace=True)
    # print(df.to_string())

    b, a = cheby2_bandpass(freq - bw/2, freq + bw/2, fs)
    w, h = signal.freqz(b, a=a)

    fig, ax1 = plt.subplots()
    ax1.set_title('Digital filter frequency response')
    ax1.plot(w*fs/2/np.pi, 20 * np.log10(abs(h)), 'b')
    ax1.set_ylabel('Amplitude [dB]', color='b')
    ax1.set_xlabel('Frequency [rad/sample]')
    plt.show()

    # filepath = sys.argv[1]
    #
    # df = pandas.read_csv(filepath, skiprows=[1], skipinitialspace=True)
    # data1 = df["Channel 0"].tolist()
    # data2 = df["Channel 1"].tolist()
    # data3 = df["Channel 2"].tolist()
    # data4 = df["Channel 3"].tolist()
    #
    # t = np.arange(len(data1))
    #
    # sp = np.fft.fft(data1)
    # freq = np.fft.fftfreq(t.shape[-1])
    # plt.figure()
    # plt.plot(freq, np.absolute(sp))
    # plt.figure()
    # plt.plot(freq, np.angle(sp))
    # plt.show()
