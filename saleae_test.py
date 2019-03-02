<<<<<<< HEAD
import saleae
import csv
import numpy as np
from scipy.signal import cheby2, lfilter
import sys
import math
import os.path
import time
# import subprocess
#
# process = subprocess.Popen("/home/estellehe/Desktop/Logic/Logic")

#sampling frequency
fs = 625000
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
=======
import csv
import os
from scipy.signal import cheby2, lfilter

freq = 40000

bandpassw = 500

fs = 130000
>>>>>>> 3ff278573f7b578abccdd6860c60e4601478148d


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


<<<<<<< HEAD

if __name__ == "__main__":
    try:
        s = saleae.Saleae()
        time.sleep(1)
        print s.get_active_device()
    except:
        print "logic software not open or device not found"

#     s.set_active_channels([], [0, 1, 2])
#     s.set_capture_seconds(3)
#     s.set_sample_rate(s.get_all_sample_rates()[3])
#     #
#     # s.capture_start_and_wait_until_finished()
#     # s.export_data2('/home/estellehe/Desktop/Acoustics/test.csv', analog_channels=[0, 1, 2])
#
# if __name__ == "__main__":
#     #sampling
#     data0 = []
#     data1 = []
#     data2 = []
#     freq = 40000
#     # angle = int(sys.argv[1])
#     #calling samping program, get result from stdin stdout
#     process = subprocess.Popen(["/home/estellehe/Desktop/Linux_Drivers/USB/mcc-libusb/sampling", str(ts), str(fs)], stdout = subprocess.PIPE)
#     stddata, stderror = process.communicate()
#     if "fail" in stddata:
#         print stddata
#         sys.exit()
#
#     #parse data from stdin
#     datas = stddata.split("\n")
#     for d in datas:
# 	ds = d.split(",")
#         try:
#             data0.append(float(ds[1]))
#             data1.append(float(ds[2]))
#             data2.append(float(ds[0]))
#         except:
#             continue
#
#     sample_count = 1
#     filename = "_40k_130k.csv"
#     # filename = "_{}_130k_40k_{}.csv".format(angle, sample_count)
#     data_path = "/home/estellehe/Desktop/Data/"
#     # while os.path.isfile(data_path+"data"+filename):
#     #     sample_count += 1
#     #     filename = "_{}_130k_40k_{}.csv".format(angle, sample_count)
#
#     with open(data_path+"data"+filename, 'wb') as write:
#         writer = csv.writer(write)
#         for k in range(len(data0)):
#             writer.writerow([round(data0[k], 4), round(data1[k], 4), round(data2[k], 4)])
#     #bandpass
#     try:
#         out0 = cheby2_bandpass_filter(data0, freq-bandpassw/2, freq+bandpassw/2, fs)
#         out1 = cheby2_bandpass_filter(data1, freq-bandpassw/2, freq+bandpassw/2, fs)
#         out2 = cheby2_bandpass_filter(data2, freq-bandpassw/2, freq+bandpassw/2, fs)
#     except Exception as e:
#         print(e)
#
#     with open(data_path+"out"+filename, 'wb') as write:
#         writer = csv.writer(write)
#         for k in range(len(out0)):
#             writer.writerow([round(out0[k], 4), round(out1[k], 4), round(out2[k], 4)])
=======
if __name__ == "__main__":
    #sampling
    datapath = "/Users/estellehe/Documents/Robo/Acoustic/data/"
    outpath = "/Users/estellehe/Documents/Robo/Acoustic/filtered_data/"
    file = "data_40k_130k.csv"

    data0 = []
    out0 = []

    with open(os.path.join(datapath, file), 'rb') as filec:
        reader = csv.reader(filec)
        for row in reader:
            try:
                data0.append(float(row[2]))
            except:
                continue

    try:
        out0 = cheby2_bandpass_filter(data0, freq-bandpassw/2, freq+bandpassw/2, fs)
    except Exception as e:
        print(e)

    with open(os.path.join(outpath, file.replace(".csv", "_filtered.csv")), 'wb') as write:
        writer = csv.writer(write)
        for k in range(len(out0)):
            writer.writerow([round(out0[k], 4)])
>>>>>>> 3ff278573f7b578abccdd6860c60e4601478148d
