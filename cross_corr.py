from scipy.interpolate import interp1d
from scipy.signal import correlate
import matplotlib.pyplot as plt
import numpy as np
import csv

#sampling frequency
fs = 130000
#sampling time period
ts = 3
#number of sample taken during the ping
pingc = fs*0.004

def find_first_window(out, avem):
    start = 0
    end = len(out) - 1
    if avem-int(pingc*6) > start:
        start = avem-int(pingc*6)
        start = int(start)
    if avem+int(pingc) < end:
        end = avem+int(pingc)
        end = int(end)
    return start, end

def find_second_window(outw, aved):
    starts = 0
    ends = len(outw)-1
    if aved-int(pingc/50) > starts:
        starts = aved-int(pingc/50)
        starts = int(starts)
    if aved+int(pingc/20) < ends:
        ends = aved+int(pingc/20)
        ends = int(ends)
    return starts, ends

def moving_average_double(a, n = pingc/10):
    weights = np.repeat(1.0, n)/n
    alist = np.convolve(a, weights, 'valid')
    lasta = alist[0]
    lastd = 0
    for k in range(len(alist)):
    	ave = alist[k]
        currentd = ave-lasta
        if currentd > lastd:
            if currentd > lastd*2 and lastd != 0:
                return int(k+n)
            lastd = currentd
        lasta = ave
    return 0

def moving_average_max(a, n = pingc) :
    weights = np.repeat(1.0, n)/n
    alist = np.convolve(a, weights, 'valid')
    #ret = np.cumsum(a, dtype=float)
    #ret[n:] = ret[n:] - ret[:-n]
    #alist = ret[n - 1:] / n
    # aglist = np.gradient(alist)
    # print aglist
    # for k in range(len(aglist)):
    #     ave = aglist[k]
    #     if ave < 0:
    #         return k
    maxa = 0
    maxi = 0
    for k in range(len(alist)):
    	ave = alist[k]
    	if ave > maxa:
    	    maxa = ave
    	    maxi = k
    return maxi

if __name__ == "__main__":

    data0 = []
    data1 = []
    data2 = []
    filepath = "/Users/estellehe/Documents/Robo/Acoustic/filtered_data/data_0_130k_40k_1_filtered.csv"

    with open(filepath, 'rb') as filec:
        reader = csv.reader(filec)
        for row in reader:
            try:
                data0.append(float(row[0]))
                data1.append(float(row[1]))
                data2.append(float(row[2]))
            except:
                continue

    out0 = data0[13000:]
    out1 = data1[13000:]
    out2 = data2[13000:]
    out = out0 + out1 + out2

    #find rough ping window with moving_average_max
    outsq0 = np.absolute(out0)
    outsq1 = np.absolute(out1)
    outsq2 = np.absolute(out2)
    outsq = outsq0+outsq1+outsq2

    avem = moving_average_max(outsq)

    start, end = find_first_window(out, avem)

    # print "avem, start, end", avem, start, end
    outw0 = out0[start:(end+1)]
    outw1 = out1[start:(end+1)]
    outw2 = out2[start:(end+1)]
    outw = outsq[start:(end+1)]
    #dataw = data[start:(end+1)]
    # time = np.linspace(start/fs, end/fs, end-start+1)
    # mag = np.sum(np.absolute(out))
    # print "magnitude", mag

    #use moving_average_double to locate the very first part
    aved = moving_average_double(np.absolute(outw))

    starts, ends = find_second_window(outw, aved)

    print aved, starts, ends

    # print "aved, start, end", aved, starts, ends
    outsw0 = outw0[starts:(ends+1)]
    outsw1 = outw1[starts:(ends+1)]
    outsw2 = outw2[starts:(ends+1)]
    outsw = outw[starts:(ends+1)]





    time = np.linspace(0, len(outsw0)-1 ,len(outsw0)*10 - 9)

    data0f = interp1d(range(0, len(outsw0)), outsw0, kind = 'cubic')
    data1f = interp1d(range(0, len(outsw1)), outsw1, kind = 'cubic')
    data2f = interp1d(range(0, len(outsw1)), outsw2, kind = 'cubic')

    data0_intp = data0f(time)
    data1_intp = data1f(time)
    data2_intp = data2f(time)

    xcrr_x = correlate(data0_intp, data1_intp, mode = 'full')
    xcrr_y = correlate(data0_intp, data2_intp, mode = 'full')

    max_loc_x = np.argmax(xcrr_x)
    max_loc_y = np.argmax(xcrr_y)

    plt.figure()
    plt.plot(outsw0)
    plt.plot(outsw1)
    plt.plot(outsw2)
    plt.show()
