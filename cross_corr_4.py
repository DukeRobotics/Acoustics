import pandas
from scipy.signal import correlate
import numpy as np
import math
import sys

#sampling frequency
fs = 625000
#sampling time period
ts = 3
#number of sample taken during the ping
pingc = fs*0.004
#target frequency
freq = 40000
#speed of sound in water
vsound = 1481
#nipple distance between hydrophone
spac = 0.012
#allowed phase diff
dphase = math.pi*2/(vsound/freq)*spac


def moving_average_increase(a, n = math.ceil(fs/freq)):
    weights = np.repeat(1.0, n)/n
    alist = np.convolve(a, weights, 'valid')
    lasta = alist[0]
    counter = 0
    for k in range(len(alist)/n):
    	ave = alist[k*n]
        if ave > lasta:
            counter += 1
        if ave <= lasta:
            counter = 0
        if counter >= 10:
            return int((k-10)*n)
        lasta = ave
    return 0

def find_first_window(data, avem):
    start = 0
    end = len(out) - 1
    if avem-int(pingc*6) > start:
        start = avem-int(pingc*6)
        start = int(start)
    if avem+int(pingc) < end:
        end = avem+int(pingc)
        end = int(end)
    return start, end

def moving_average_max(a, n = pingc) :
    weights = np.repeat(1.0, n)/n
    alist = np.convolve(a, weights, 'valid')
    maxa = 0
    maxi = 0
    for k in range(len(alist)):
    	ave = alist[k]
    	if ave > maxa:
    	    maxa = ave
    	    maxi = k
    return maxi

if __name__ == "__main__":

    filepath = sys.argv[1]

    df = pandas.read_csv(filepath, skiprows=[1], skipinitialspace=True)
    data1 = df["Channel 0"].tolist()
    data2 = df["Channel 1"].tolist()
    data3 = df["Channel 2"].tolist()
    data4 = df["Channel 3"].tolist()

    datasq1 = np.absolute(data1)
    datasq2 = np.absolute(data2)
    datasq3 = np.absolute(data3)
    datasq4 = np.absolute(data4)
    #since this is a rough window so we do only one max moving ave on the sum with length of ping
    datasq = datasq1+datasq2+datasq3+datasq4

    avem = moving_average_max(datasq)

    start, end = find_first_window(datasq, avem)

    dataw1 = datasq1[start:(end+1)]
    dataw2 = datasq2[start:(end+1)]
    dataw3 = datasq3[start:(end+1)]
    dataw4 = datasq4[start:(end+1)]

    
