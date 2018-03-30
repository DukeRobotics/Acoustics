from scipy.signal import cheby2, lfilter
import csv
import matplotlib.pyplot as plt
import subprocess
import numpy as np
import sys
import math

fs = 130000

ts = 3
pingc = fs*0.004
vsound = 1481
spac = 0.016
cycle = 1/float(fs)

#running average get time section, fft get phase comparison, multichannels

def cheby2_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = cheby2(order, 5, [low, high], btype='bandpass')
    return b, a

def cheby2_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = cheby2_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def phase_diff(late, early):
    if late > early:
        return late - early
    else:
        return late - (early - cycle)

def moving_average(a, n = pingc*3) :
    weights = np.repeat(1.0, n)/n
    alist = np.convolve(a, weights, 'valid')
    #ret = np.cumsum(a, dtype=float)
    #ret[n:] = ret[n:] - ret[:-n]
    #alist = ret[n - 1:] / n
    maxa = 0
    maxi = 0
    for k in range(len(alist)):
    	ave = alist[k]
    	if ave > maxa:
    	    maxa = ave
    	    maxi = k
    return maxi

if __name__ == "__main__":
    #sampling
    data0 = []
    data1 = []
    data2 = []
    freq = int(sys.argv[1])
    process = subprocess.Popen(["/home/robot/Desktop/mcc-libusb/sampling", str(ts), str(fs)], stdout = subprocess.PIPE)
    stddata, stderror = process.communicate()

    #parse data from stdin
    datas = stddata.split("\n")
    for d in datas:
	ds = d.split(",")
        try:
            data0.append(float(ds[0]))
            data1.append(float(ds[1]))
            data2.append(float(ds[2]))
        except:
            continue
    data0 = data0[13000:]
    data1 = data1[13000:]
    data2 = data2[13000:]

    # with open("data.csv", 'rb') as filec:
    #     reader = csv.reader(filec)
    #     for row in reader:
    #         try:
    #             p = float(row[0])
	# 	#print p
    #             data.append(p)
    #         except:
    #             continue

    #bandpass
    try:
        out0 = cheby2_bandpass_filter(data0, freq-250, freq+250, fs)
        out1 = cheby2_bandpass_filter(data1, freq-250, freq+250, fs)
        out2 = cheby2_bandpass_filter(data2, freq-250, freq+250, fs)
    except Exception as e:
        print(e)

    #find window with moving_average
    out = out0+out1+out2
    avem = moving_average(out)
    start = 0
    end = len(out) - 1
    if avem-pingc*6 > start:
        start = avem-pingc*6
        start = int(start)
    if avem+pingc*6 < end:
        end = avem+pingc*6
        end = int(end)
    print avem, start, end

    startT = start/fs
    endT = end/fs
