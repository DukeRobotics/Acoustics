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
    outw0 = out0[start:(end+1)]
    outw1 = out1[start:(end+1)]
    outw2 = out2[start:(end+1)]
    outw = out[start:(end+1)]
    #dataw = data[start:(end+1)]
    time = np.linspace(start/fs, end/fs, end-start+1)

    #fft
    fft0 = np.fft.fft(outw0)
    ffta0 = np.absolute(fft0)
    fft1 = np.fft.fft(outw1)
    ffta1 = np.absolute(fft1)
    fft2 = np.fft.fft(outw2)
    ffta2 = np.absolute(fft2)
    fft = np.fft.fft(outw)
    ffta = np.absolute(fft)
    result = 0
    resultc = 0
    resultf = 0
    resulti = 0

    #find max
    timew = (end-start+1)/float(fs)
    print timew
    for i in range(int(len(ffta)/2)):
        f = i/timew;
        if ffta[i] > result:
            resultf = f
            resulti = i
            result = ffta[i]
	    resultc = i
        #print fft[i], ffta[i], f

    resultp0 = np.imag(fft0[resultc])/np.real(fft0[resultc])/(2 * math.pi * resultf)
    resultp1 = np.imag(fft1[resultc])/np.real(fft1[resultc])/(2 * math.pi * resultf)
    resultp2 = np.imag(fft2[resultc])/np.real(fft2[resultc])/(2 * math.pi * resultf)
    cycle = 1/float(fs)
    #timediff = phase/(2pi*freq)
    print result, resultf
    print resultp0, resultp1, resultp2, cycle
    #print out[0]
    with open("out.csv", 'wb') as write:
        writer = csv.writer(write)
        for point in outw:
            writer.writerow([round(point, 4)])
    plt.plot(outw)
    plt.plot(outw0)
    plt.plot(outw1)
    plt.plot(outw2)
    plt.show()
    # print len(data)
    #print out
    # subprocess.call(["rm", "testcsv"])
    # subprocess.call(["gcc", "-o", "testcsv", "testcsv.c", "-lfftw3", "-lm"])
