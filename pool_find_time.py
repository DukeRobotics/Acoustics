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
pingc = = fs*0.004
bandpassw = 1600
t_cycle = 2.048
output_dir = "/home/robot/Documents/output/"
filter_output_dir = "/home/robot/Documents/output/filtered/"

#get bandpass filter parameter
#bandwidth need to be 800*2
def cheby2_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = cheby2(order, 60, [low, high], btype='bandpass')
    return b, a

#filter the data with bandpass
def cheby2_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = cheby2_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

#find the max moving average of the filtered data
#try to determine a rough ping window
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
    freq = 40000
    try:
        s = saleae.Saleae()
    except:
        subprocess.Popen(["/home/robot/Logic/Logic", "-socket"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(15)
        s = saleae.Saleae()
        print("saleae software down, open saleae software before next script run")
        #print("software not opened or no device detected")
        #exit()

    #todo: add a while loop here to keep trying on get_active_device, timeout after 10s
    try:
        if s.get_active_device().active:
            s.set_active_channels([], [0, 1, 2, 3])
            s.set_capture_seconds(3)
            s.set_sample_rate(s.get_all_sample_rates()[sampling_rate])
    except:
        exit()

    # try:
    #     file = open(os.path.join(output_dir, "3s", ".csv"), 'r')
    # except IOError:
    #     file = open(os.path.join(output_dir, "3s", ".csv"), 'w')

    ts = False
    ts = input("sample length, sample delay: ").split(' ')
    while ts:
        s_len = float(ts[0])
        s_delay = float(ts[1])
        bw = bandpassw

        # initial 3 second sampling
        s.set_capture_seconds(3)
        t_3s = time.time()
        s.capture_start_and_wait_until_finished()
        s.export_data2(os.path.join(output_dir, "3s_"+str(t_3s).replace('.', '_')+".csv"), analog_channels=[0, 1, 2, 3])
        t_3s_f = time.time()
        print("3 second starts at "+str(t_3s)+" and finishs at "+str(t_3s_f))

        time.sleep(4)
        # import data from csv
        df = pandas.read_csv(os.path.join(output_dir, "3s_"+str(t_3s).replace('.', '_')+".csv"), skiprows=[1], skipinitialspace=True)
        data1 = df["Channel 0"].tolist()
        data2 = df["Channel 1"].tolist()
        data3 = df["Channel 2"].tolist()
        data4 = df["Channel 3"].tolist()
        print("import length "+str(len(data1)))

        #bandpass
        try:
            out1 = cheby2_bandpass_filter(data1, freq-bw/2, freq+bw/2, fs)
            out2 = cheby2_bandpass_filter(data2, freq-bw/2, freq+bw/2, fs)
            out3 = cheby2_bandpass_filter(data3, freq-bw/2, freq+bw/2, fs)
            out4 = cheby2_bandpass_filter(data4, freq-bw/2, freq+bw/2, fs)
        except Exception as e:
            print(e)

        print("filtered length "+str(len(out1)))

        df = pandas.DataFrame()
        df["out1"] = out1;
        df["out2"] = out2;
        df["out3"] = out3;
        df["out4"] = out4;
        df.to_csv(os.path.join(filter_output_dir, "3s_"+str(t_3s).replace('.', '_')+"_filtered.csv"))

        #find first front with moving_average_max
        outsq1 = np.absolute(out1[150:])
        outsq2 = np.absolute(out2[150:])
        outsq3 = np.absolute(out3[150:])
        outsq4 = np.absolute(out4[150:])
        avem1 = moving_average_max(outsq1)+150
        avem2 = moving_average_max(outsq2)+150
        avem3 = moving_average_max(outsq3)+150
        avem4 = moving_average_max(outsq4)+150
        first_front = min(avem1, avem2, avem3, avem4)/fs + t_3s
        max_3s = max(outsq1+outsq2+outsq3+outsq4)

        r_mag = 0
        r_index = 0

        # every adjustment of sampling period move the offset 0.01s forward
        # want the max stay in the second fourth of sampling duration(0.08s)
        while r_mag < 2.0/3.0 or r_index < 1.0/4.0 or r_index > 1.0/2.0:

            if r_mag != 0:
                if r_mag < 2.0/3.0:
                    s_delay = s_delay + 0.01
                else:
                    if r_index < 1.0/4.0:
                        s_delay = s_delay + 0.01
                    elif r_index > 1.0/2.0:
                        s_delay = s_delay - 0.01
            print("\ncurrent delay is "+s_delay)
            # do 0.08 second sampling, 0.02 second before the first first_front, so offset = -0.02
            # get next cycle start time
            cycle = 0
            while cycle*t_cycle+first_front-s_delay < time.time():
                cycle = cycle + 1
            next_start = cycle*t_cycle+first_front-s_delay
            latest_front = cycle*t_cycle+first_front
            # wait for start time and do 0.5 second sampling
            while time.time() < next_start:
                pass
            s.set_capture_seconds(s_len)
            t_1s_1 = time.time()
            s.capture_start_and_wait_until_finished()
            s.export_data2(os.path.join(output_dir, "1s_"+str(t_1s_1).replace('.', '_')+".csv"), analog_channels=[0, 1, 2, 3])
            t_1s_1_f = time.time()
            print("1 second starts at "+str(t_1s_1)+" and finishs at "+str(t_1s_1_f))


            time.sleep(1)
            # import data from csv
            df = pandas.read_csv(os.path.join(output_dir, "1s_"+str(t_1s_1).replace('.', '_')+".csv"), skiprows=[1], skipinitialspace=True)
            data1 = df["Channel 0"].tolist()
            data2 = df["Channel 1"].tolist()
            data3 = df["Channel 2"].tolist()
            data4 = df["Channel 3"].tolist()
            print("import length "+str(len(data1)))

            #bandpass
            try:
                out1 = cheby2_bandpass_filter(data1, freq-bw/2, freq+bw/2, fs)
                out2 = cheby2_bandpass_filter(data2, freq-bw/2, freq+bw/2, fs)
                out3 = cheby2_bandpass_filter(data3, freq-bw/2, freq+bw/2, fs)
                out4 = cheby2_bandpass_filter(data4, freq-bw/2, freq+bw/2, fs)
            except Exception as e:
                print(e)

            print("filtered length "+str(len(out1)))

            df = pandas.DataFrame()
            df["out1"] = out1;
            df["out2"] = out2;
            df["out3"] = out3;
            df["out4"] = out4;
            df.to_csv(os.path.join(filter_output_dir, "1s_"+str(t_1s_1).replace('.', '_')+"_filtered.csv"))

            #find first front with moving_average_max
            outsq1 = np.absolute(out1[150:])
            outsq2 = np.absolute(out2[150:])
            outsq3 = np.absolute(out3[150:])
            outsq4 = np.absolute(out4[150:])
            total = outsq1+outsq2+outsq3+outsq4
            max_1s = max(total)
            index = np.argmax(total)+150
            print("\nmax 3s is "+str(max_3s)+" and max 1s is "+str(max_1s)+" at index "+str(index))

            r_mag = max_1s/max_3s
            r_index = index/(s_len*fs)
            print("1s-to-3s magnitude ratio is "+str(r_mag))
            print("max index ratio is "+str(r_index))


        ts = False
        ts = input("sample length, sample delay: ").split(' ')
