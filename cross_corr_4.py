import pandas
# correlate(in1, in2) = k: in2 faster than in1 by k
from scipy.signal import correlate, cheby2, lfilter
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
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

bw = 1600

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


def diff_equation(hp1, hp2, target, t_diff):
    return (np.linalg.norm(target-hp1) - np.linalg.norm(target-hp2)) - t_diff

def system(target, *data):
    hp1, hp2, hp3, hp4, diff_12, diff_23, diff_34= data
    return (diff_equation(hp1, hp2, target, diff_12), diff_equation(hp2, hp3, target, diff_23), diff_equation(hp3, hp4, target, diff_34))

# need to adjust n and threshold for different data set
def moving_average_increase(a, n = math.ceil(fs/freq)):
    weights = np.repeat(1.0, n)/n
    alist = np.convolve(a, weights, 'valid')
    # even more smooth
    alist = np.convolve(alist, weights, 'valid')
    lasta = alist[0]
    start_o = 0
    end_o = 0
    start = 0
    end = 0
    last_inc = False
    for k in range(len(alist)):
        ave = alist[k]
        if last_inc:
            if ave > lasta:
                end = k
            else:
                last_inc = False
            if (alist[end] - alist[start]) > (alist[end_o] - alist[start_o]):
                start_o = start
                end_o = end
        elif ave > lasta:
            start = k
            last_inc = True
        # if ave > lasta:
        #     end = k
        # if ave <= lasta and :
        #     start = k
        # # adjust this when necessary
        # if counter > 500 and ave+0.0001 > (alist[k-500]+0.0001)*5:
        #     # print(int(k-500))
        #     # plt.figure()
        #     # plt.plot(alist)
        #     # plt.show()
        #     return int(k-500)
        lasta = ave
    return start_o, end_o

def find_first_window(data, avem):
    start = 0
    end = len(data) - 1
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

def find_second_window(outw, aved):
    starts = 0
    ends = len(outw)-1
    if aved+math.ceil(fs/freq) < ends:
        starts = aved+math.ceil(fs/freq)
        starts = int(starts)
    # might need to change window size
    if aved+math.ceil(fs/freq)*5 < ends:
        ends = aved+math.ceil(fs/freq)*5
        ends = int(ends)
    return starts, ends

if __name__ == "__main__":

    filepath = sys.argv[1]

    df = pandas.read_csv(filepath, skiprows=[1], skipinitialspace=True)
    data1 = df["Channel 0"].tolist()
    data2 = df["Channel 1"].tolist()
    data3 = df["Channel 2"].tolist()
    data4 = df["Channel 3"].tolist()

    dataf1 = cheby2_bandpass_filter(data1, freq-bw/2, freq+bw/2, fs)
    dataf2 = cheby2_bandpass_filter(data2, freq-bw/2, freq+bw/2, fs)
    dataf3 = cheby2_bandpass_filter(data3, freq-bw/2, freq+bw/2, fs)
    dataf4 = cheby2_bandpass_filter(data4, freq-bw/2, freq+bw/2, fs)



    datasq1 = np.absolute(dataf1[13000:])
    datasq2 = np.absolute(dataf2[13000:])
    datasq3 = np.absolute(dataf3[13000:])
    datasq4 = np.absolute(dataf4[13000:])
    #since this is a rough window so we do only one max moving ave on the sum with length of ping
    datasq = datasq1+datasq2+datasq3+datasq4

    avem = moving_average_max(datasq)
    # print(avem+13000)

    start, end = find_first_window(datasq, avem)
    # print(start+13000, end+13000)


    dataw1 = datasq1[start:(end+1)]
    dataw2 = datasq2[start:(end+1)]
    dataw3 = datasq3[start:(end+1)]
    dataw4 = datasq4[start:(end+1)]
    dataw = datasq[start:(end+1)]

    starts1, ends1 = moving_average_increase(dataw1)
    starts2, ends2 = moving_average_increase(dataw2)
    starts3, ends3 = moving_average_increase(dataw3)
    starts4, ends4 = moving_average_increase(dataw4)
    starts = max([starts1, starts2, starts3, starts4])
    ends = min([ends1, ends2, ends3, ends4])
    # print(starts, ends)


    # #use moving_average_double to locate the very first part
    # aved1 = moving_average_increase(dataw1)
    # aved2 = moving_average_increase(dataw2)
    # aved3 = moving_average_increase(dataw3)
    # aved4 = moving_average_increase(dataw4)
    # aved = max([aved1, aved2, aved3, aved4])
    # print(aved1, aved2, aved3, aved4)
    #
    # starts, ends = find_second_window(dataw, aved)
    # print(start+starts+13000, start+ends+13000)

    # plt.figure()
    # plt.plot(np.absolute(dataw))
    # plt.figure()
    # plt.plot(dataw1)
    # plt.plot(dataw2)
    # plt.plot(dataw3)
    # plt.plot(dataw4)
    # plt.legend(['1', '2', '3', '4'])
    # plt.show()


    # # print "aved, start, end", aved, starts, ends
    datasw1 = dataw1[starts:(ends+1)]
    datasw2 = dataw2[starts:(ends+1)]
    datasw3 = dataw3[starts:(ends+1)]
    datasw4 = dataw4[starts:(ends+1)]

# not sure if we really need the interpolation

    # 10 times sampling points, so 10 times sampling frequency
    time = np.linspace(0, len(datasw1)-1 ,len(datasw1)*10 - 9)

    data1f = interp1d(range(0, len(datasw1)), datasw1, kind = 'cubic')
    data2f = interp1d(range(0, len(datasw2)), datasw2, kind = 'cubic')
    data3f = interp1d(range(0, len(datasw3)), datasw3, kind = 'cubic')
    data4f = interp1d(range(0, len(datasw4)), datasw4, kind = 'cubic')

    data1_intp = data1f(time)
    data2_intp = data2f(time)
    data3_intp = data3f(time)
    data4_intp = data4f(time)

# only use the middle section to avoid distortion

    intp_len = math.ceil(len(data1_intp)/4)
    data1_intpw = data1_intp[intp_len*2: intp_len*3]
    data2_intpw = data2_intp[intp_len*2: intp_len*3]
    data3_intpw = data3_intp[intp_len*2: intp_len*3]
    data4_intpw = data4_intp[intp_len*2: intp_len*3]

# cross correlation

    xcrr_12 = correlate(data1_intpw, data2_intpw, mode = 'full')
    xcrr_23 = correlate(data2_intpw, data3_intpw, mode = 'full')
    xcrr_34 = correlate(data3_intpw, data4_intpw, mode = 'full')

# when max_xcrr is left to the center, in1 is left to in2, in1 earlier than in2, negative p_diff
    diff_12 = (np.argmax(xcrr_12) - len(data1_intpw) + 1)/(10*fs)*vsound
    diff_23 = (np.argmax(xcrr_23) - len(data1_intpw) + 1)/(10*fs)*vsound
    diff_34 = (np.argmax(xcrr_34) - len(data1_intpw) + 1)/(10*fs)*vsound
    print(diff_12, diff_23, diff_34)

    # plt.figure()
    # plt.plot(data1_intpw)
    # plt.plot(data2_intpw)
    # plt.plot(data3_intpw)
    # plt.plot(data4_intpw)
    # plt.legend(['1', '2', '3', '4'])
    # plt.show()

    hp1 = np.array([0, 0, 0])
    hp2 = np.array([0, -spac, 0])
    hp3 = np.array([-spac, 0, 0])
    hp4 = np.array([-spac, -spac, 0])
    # target = np.array([-1, -1, -1])
    # diff_12 = np.linalg.norm(target-hp1) - np.linalg.norm(target-hp2)
    # diff_23 = np.linalg.norm(target-hp2) - np.linalg.norm(target-hp3)
    # diff_34 = np.linalg.norm(target-hp3) - np.linalg.norm(target-hp4)


    data_val = (hp1, hp2, hp3, hp4, diff_12, diff_23, diff_34)
    solution = fsolve(system, (0, 0, 0), args=data_val)
    print(solution)
