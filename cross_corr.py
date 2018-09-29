from scipy.interpolate import interp1d
from scipy.signal import correlate
import numpy as np
import csv

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


time = np.linspace(0, len(data0)-1 ,len(data0)*10 - 9)

data0f = interp1d(range(0, len(data0)), data0, kind = 'cubic')
data1f = interp1d(range(0, len(data1)), data1, kind = 'cubic')
data2f = interp1d(range(0, len(data2)), data2, kind = 'cubic')

data0_intp = data0f(time)
data1_intp = data1f(time)
data2_intp = data2f(time)

xcrr_x = correlate(data0_intp, data1_intp, mode = 'full')
xcrr_y = correlate(data0_intp, data2_intp, mode = 'full')

max_loc_x = np.argmax(xcrr_x)
max_loc_y = np.argmax(xcrr_y)

print len(xcrr_x), len(data0_intp), len(data0)
