import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import cheby2, lfilter
from scipy.optimize import leastsq
import sys
import math
import subprocess
import os.path

#sampling frequency
fs = 130000
#sampling time period
ts = 3
#number of sample taken during the ping
pingc = fs*0.004
#speed of sound in water
vsound = 1481
#nipple distance between hydrophone
spac = 0.01275
#time for one sample
cycle = 1/float(fs)
#window for bandpass
bandpassw = 500
#when using fft, the allowed window for frequency
fftfreqw = 500


#get bandpass filter parameter
def cheby2_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = cheby2(order, 5, [low, high], btype='bandpass')
    return b, a

#filter the data with bandpass
def cheby2_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = cheby2_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

if __name__ == "__main__":
    #sampling
    data0 = []
    data1 = []
    data2 = []
    freq = 40000
    angle = int(sys.argv[1])
    #calling samping program, get result from stdin stdout
    process = subprocess.Popen(["/home/estellehe/Desktop/Linux_Drivers/USB/mcc-libusb/sampling", str(ts), str(fs)], stdout = subprocess.PIPE)
    stddata, stderror = process.communicate()
    if "fail" in stddata:
        print stddata
        sys.exit()

    #parse data from stdin
    datas = stddata.split("\n")
    for d in datas:
	ds = d.split(",")
        try:
            data0.append(float(ds[1]))
            data1.append(float(ds[2]))
            data2.append(float(ds[0]))
        except:
            continue

    sample_count = 1
    filename = "_{}_130k_40k_{}.csv".format(angle, sample_count)
    data_path = ""
    while os.path.isfile(data_path+"data"+filename):
        sample_count += 1
        filename = "_{}_130k_40k_{}.csv".format(angle, sample_count)

    with open(data_path+"data"+filename, 'wb') as write:
        writer = csv.writer(write)
        for k in range(len(data0)):
            writer.writerow([round(data0[k], 4), round(data1[k], 4), round(data2[k], 4)])
    #bandpass
    try:
        out0 = cheby2_bandpass_filter(data0, freq-bandpassw/2, freq+bandpassw/2, fs)
        out1 = cheby2_bandpass_filter(data1, freq-bandpassw/2, freq+bandpassw/2, fs)
        out2 = cheby2_bandpass_filter(data2, freq-bandpassw/2, freq+bandpassw/2, fs)
    except Exception as e:
        print(e)

    with open(data_path+"out"+filename, 'wb') as write:
        writer = csv.writer(write)
        for k in range(len(out)):
            writer.writerow([round(out0[k], 4), round(out1[k], 4), round(out2[k], 4)])
