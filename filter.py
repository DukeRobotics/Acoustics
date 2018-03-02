from scipy.signal import cheby2, lfilter
import csv
import matplotlib.pyplot as plt
import subprocess
import numpy as np
import sys
import math

fs = 130000

ts = 2.15
pingc = fs*0.004

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
    data = []
    freq = int(sys.argv[1])
    process = subprocess.Popen(["/home/robot/Desktop/mcc-libusb/sampling", str(ts), str(fs)], stdout = subprocess.PIPE)
    stddata, stderror = process.communicate()

    #parse data from stdin
    datas = stddata.split("\n")
    for d in datas:
        try:
            p = float(d)
            data.append(p)
        except:
            continue
    data = data[13000:]

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
        out = cheby2_bandpass_filter(data, freq-50, freq+50, fs)
    except Exception as e:
        print(e)

    #find window with moving_average
    avem = moving_average(out)
    start = 0
    end = len(out) - 1
    if avem-pingc*6 > start:
        start = avem-pingc*6
        start = int(start)
    if avem+pingc*6 < end:
        end = avem+pingc*6
        end = int(end)
    print start, end
    outw = out[start:(end+1)]
    dataw = data[start:(end+1)]
    time = np.linspace(start/fs, end/fs, end-start+1)

    #fft
    fft = np.fft.fft(dataw)
    ffta = np.absolute(fft)
    result = 0
    resultf = 0
    resulti = 0

    #find max
    timew = (end-start+1)/float(fs)
    print timew
    for i in range(len(ffta)):
        f = i/timew;
        if ffta[i] > result:
            resultf = f
            resulti = i
            result = ffta[i]
        #print fft[i], ffta[i], f

    resultp = np.imag(result)/np.real(result)/(2 * math.pi * resultf)
    #timediff = phase/(2pi*freq)
    print result, resultf, resultp
    #print out[0]
    with open("out.csv", 'wb') as write:
        writer = csv.writer(write)
        for point in outw:
            writer.writerow([round(point, 4)])
    with open("data.csv", 'wb') as write:
        writer = csv.writer(write)
        for point in dataw:
            writer.writerow([round(point, 4)])
    #plt.plot(dataw)
    plt.plot(outw)
    plt.show()
    # print len(data)
    #print out
    # subprocess.call(["rm", "testcsv"])
    # subprocess.call(["gcc", "-o", "testcsv", "testcsv.c", "-lfftw3", "-lm"])
