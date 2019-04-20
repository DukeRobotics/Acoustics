from __future__ import division
import saleae
import subprocess
import time
import os.path
import sys
import pandas
import numpy as np
from scipy.signal import cheby2, lfilter


# each ping is 0.04 second long
# each ping is 2.048 seconds apart
# 2 = 1250 kS/s, 3 = 625 kS/s, 4 = 125 kS/s
sampling_rate = 3
fs = 625000
pingc = pingc = fs*0.004
bandpassw = 500
t_cycle = 2.048
output_dir = "/home/estellehe/Desktop/output/"

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

#find the max moving average of the filtered data
#try to determine a rough ping window
def moving_average_max(a, n = pingc):
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

def wave_to_period_sum(a, period):
    split_index = (np.arange(np.ceil(len(a)/period))+1)*period
     

if __name__ == "__main__":
    freq = 40000

    # import data from csv
    df = pandas.read_csv(os.path.join(output_dir, "3s_"+str(t_3s).replace('.', '_')+".csv"), skiprows=[1], skipinitialspace=True)
    data1 = df["Channel 0"].tolist()[1:]
    data2 = df["Channel 1"].tolist()[1:]
    data3 = df["Channel 2"].tolist()[1:]
    data4 = df["Channel 3"].tolist()[1:]

    #bandpass
    try:
        out1 = cheby2_bandpass_filter(data1, freq-bandpassw/2, freq+bandpassw/2, fs)
        out2 = cheby2_bandpass_filter(data2, freq-bandpassw/2, freq+bandpassw/2, fs)
        out3 = cheby2_bandpass_filter(data3, freq-bandpassw/2, freq+bandpassw/2, fs)
        out4 = cheby2_bandpass_filter(data4, freq-bandpassw/2, freq+bandpassw/2, fs)
    except Exception as e:
        print(e)

    #find first front with moving_average_max
    outsq1 = np.absolute(out1)
    outsq2 = np.absolute(out2)
    outsq3 = np.absolute(out3)
    outsq4 = np.absolute(out4)
    max = max(max(outsq1), max(outsq2), max(outsq3), max(outsq4))
    avem1 = moving_average_max(outsq1)
    avem2 = moving_average_max(outsq2)
    avem3 = moving_average_max(outsq3)
    avem4 = moving_average_max(outsq4)
    first_front = min(avem1, avem2, avem3, avem4)/fs + t_3s
    max_3s = max(max(outsq1), max(outsq2), max(outsq3), max(outsq4))

    # do 0.09 second sampling, 0.02 second before the first first_front
    # get next cycle start time
    cycle = 0
    while cycle*t_cycle+first_front-float(ts[1]) < time.time():
        cycle = cycle + 1
    next_start = cycle*t_cycle+first_front-float(ts[1])
    latest_front = cycle*t_cycle+first_front
    # wait for start time and do 0.5 second sampling
    while time.time() < next_start:
        pass
    s.set_capture_seconds(float(ts[0]))
    t_1s_1 = time.time()
    s.capture_start_and_wait_until_finished()
    s.export_data2(os.path.join(output_dir, "1s_"+str(t_1s_1).replace('.', '_')+".csv"), analog_channels=[0, 1, 2, 3])
    t_1s_1_f = time.time()
    print("1 second starts at "+str(t_1s_1)+" and finishs at "+str(t_1s_1_f))

    # import data from csv
    df = pandas.read_csv(os.path.join(output_dir, "1s_"+str(t_1s_1).replace('.', '_')+".csv"), skiprows=[1], skipinitialspace=True)
    data1 = df["Channel 0"].tolist()[1:]
    data2 = df["Channel 1"].tolist()[1:]
    data3 = df["Channel 2"].tolist()[1:]
    data4 = df["Channel 3"].tolist()[1:]

    #bandpass
    try:
        out1 = cheby2_bandpass_filter(data1, freq-bandpassw/2, freq+bandpassw/2, fs)
        out2 = cheby2_bandpass_filter(data2, freq-bandpassw/2, freq+bandpassw/2, fs)
        out3 = cheby2_bandpass_filter(data3, freq-bandpassw/2, freq+bandpassw/2, fs)
        out4 = cheby2_bandpass_filter(data4, freq-bandpassw/2, freq+bandpassw/2, fs)
    except Exception as e:
        print(e)

    #find first front with moving_average_max
    outsq1 = np.absolute(out1)
    outsq2 = np.absolute(out2)
    outsq3 = np.absolute(out3)
    outsq4 = np.absolute(out4)
    max_1s = max(max(outsq1), max(outsq2), max(outsq3), max(outsq4))
    print("max 3s is "+str(max_1s)+" and max 1s is "+str(max_1s))
