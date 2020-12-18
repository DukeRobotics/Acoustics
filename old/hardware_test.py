import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import cheby2, lfilter
from scipy.optimize import leastsq
import sys
import math
import subprocess
import time

#sampling frequency
fs = 125000
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

if __name__ == "__main__":
    #sampling
    data0 = []
    data1 = []
    data2 = []
    data3 = []
    #calling samping program, get result from stdin stdout
    t1 =  time.time()
    process = subprocess.Popen(["/home/estellehe/Desktop/Linux_Drivers/USB/mcc-libusb/sampling_4", str(ts), str(fs)], stdout = subprocess.PIPE)
    stddata, stderror = process.communicate()
    if "fail" in stddata:
        print stddata
        sys.exit()
    if stderror:
        print stderror
        sys.exit()

    print time.time()-t1

    #parse data from stdin
    datas = stddata.split("\n")
    for d in datas:
	ds = d.split(",")
        try:
            data0.append(float(ds[0]))
            data1.append(float(ds[1]))
            data2.append(float(ds[2]))
            data3.append(float(ds[3]))
        except:
            continue

    with open("out.csv", 'wb') as write:
        writer = csv.writer(write)
        for k in range(len(data0)):
            writer.writerow([data0[k], data1[k], data2[k], data3[k]])
