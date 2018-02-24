from scipy.signal import cheby2, lfilter
import csv
import matplotlib.pyplot as plt
import subprocess
import numpy as np
import sys

fs = 130000

time = 2.15

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

def moving_average(a, n = fs*0.004*3) :
    weights = np.repeat(1.0, n)/n
    alist = np.convolve(a, weights, 'valid')
    #ret = np.cumsum(a, dtype=float)
    #ret[n:] = ret[n:] - ret[:-n]
    #alist = ret[n - 1:] / n
    maxa = 0
    maxi = 0
    print len(alist)
    for k in range(len(alist)):
	ave = alist[k]
	if ave > maxa:
	    maxa = ave
	    maxi = k
    return maxi

if __name__ == "__main__":
    data = []
    freq = int(sys.argv[1])
    process = subprocess.Popen(["/home/robot/Desktop/mcc-libusb/sampling", str(time), str(fs)], stdout = subprocess.PIPE)
    stddata, stderror = process.communicate()

    datas = stddata.split("\n")

    for d in datas:
        try:
            p = float(d)
            data.append(p)
        except:
            continue

    # with open("data.csv", 'rb') as filec:
    #     reader = csv.reader(filec)
    #     for row in reader:
    #         try:
    #             p = float(row[0])
	# 	#print p
    #             data.append(p)
    #         except:
    #             continue
    try:
        out = cheby2_bandpass_filter(data[13000:], freq-50, freq+50, fs)
    except Exception as e:
        print(e)
    start = moving_average(out)
    print start
    outw = out[int(start-fs*0.004*6):int(start+fs*0.004*6)]
    time = np.linspace(0, fs*0.004*10, num=len(outw))
    #print out[0]
    # with open("out.csv", 'wb') as write:
    #     writer = csv.writer(write)
    #     for point in out:
    #         writer.writerow([round(point, 4)])
    #plt.plot(data)
    # print len(data)
    #print out
    plt.plot(out)
    # subprocess.call(["rm", "testcsv"])
    # subprocess.call(["gcc", "-o", "testcsv", "testcsv.c", "-lfftw3", "-lm"])
    plt.show()
