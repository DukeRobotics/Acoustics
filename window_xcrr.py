import pandas
# correlate(in1, in2) = k: in2 faster than in1 by k
from scipy.signal import correlate, cheby2, lfilter
import scipy.signal as signal
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
freq = 25000
#speed of sound in water
vsound = 1481
#nipple distance between hydrophone
spac = 0.012
#allowed phase diff
dphase = math.ceil(spac/vsound*fs*10)+10
#sample per cycle
spc = math.ceil(fs/freq)

#bw = 1400

def check_phase(diff):
    diff_sample = np.absolute(diff)*(10*fs)/vsound
    if diff_sample < dphase:
        return diff
    elif fs*10/freq - diff_sample < dphase:
        return (fs*10/freq - diff_sample)/(10*fs)*vsound
    else:
        return None

#bandwidth need to be 800*2
def cheby2_bandpass(lowcut, highcut, fs, k, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = cheby2(order, k, [low, high], btype='bandpass')
    return b, a

#filter the data with bandpass
def cheby2_bandpass_filter(data, lowcut, highcut, fs, k, order=5):
    b, a = cheby2_bandpass(lowcut, highcut, fs, k, order=order)
    y = lfilter(b, a, data)
    return y


def diff_equation(hp1, hp2, target, t_diff):
    return (np.linalg.norm(target-hp1) - np.linalg.norm(target-hp2)) - t_diff

def system(target, *data):
    hp1, hp2, hp3, hp4, diff_12, diff_13, diff_34= data
    return (diff_equation(hp1, hp2, target, diff_12), diff_equation(hp1, hp3, target, diff_13), diff_equation(hp3, hp4, target, diff_34))

# need to adjust n and threshold for different data set
def moving_average_increase(a, isLess, n = spc*3):
    weights = np.repeat(1.0, n)/n
    alist = np.convolve(a, weights, 'valid')
    # even more smooth
    alist = np.convolve(alist, weights, 'valid')
    # alist = np.convolve(alist, weights, 'valid')
    # alist = np.convolve(alist, weights, 'valid')
    lasta = alist[0]
    start_o = 0
    end_o = 0
    inc_arr = []
    temp = []
    start = 0
    end = 0
    last_inc = False
    inc_o = 0
    diff_o = 0
    inc_c = 0
    for k in range(np.argmax(alist)+2):
        ave = alist[k]
        if last_inc:
            if ave < lasta:
                end = k
                last_inc = False
                inc_c = alist[end] - alist[start]
                if inc_c > inc_o:
                    inc_arr.append([start_o, end_o])
                    [temp.append(inc) for inc in inc_arr if (alist[inc[1]] - alist[inc[0]])*5 > inc_c]
                    inc_arr = temp
                    temp = []
                    start_o = start
                    end_o = end
                    inc_o = alist[end_o] - alist[start_o]
                    diff_o = end_o - start_o
                elif inc_c*2 > inc_o:
                    inc_arr.append([start, end])
        elif ave > lasta:
            start = k
            last_inc = True
        lasta = ave
    inc_arr.append([start_o, end_o])
    if isLess:
        start_o = start_o
    else:
        d_inc = []
        [d_inc.append((alist[inc[1]]-alist[inc[0]])/(inc[1]-inc[0])) for inc in inc_arr]
        max_d = max(d_inc)
        temp = []
        for i in range(len(d_inc)):
            if d_inc[i]*3 > max_d:
                temp.append(inc_arr[i])
        start_o = len(alist)
        end_o = 0
        for inc in temp:
            if inc[0] < start_o:
                start_o = inc[0]
                end_o = inc[1]
        print("inc_arr", inc_arr)
        print("diff", d_inc)
        print("temp", temp)
    print(start_o, end_o)
    plt.figure()
    plt.plot(a)
    plt.figure()
    plt.plot(alist)
    plt.show()
    return start_o, end_o

def find_first_window(data, avem):
    start = 0
    end = len(data) - 1
    if avem-int(pingc*3) > start:
        start = avem-int(pingc*3)
        start = int(start)
    if avem+int(pingc*3) < end:
        end = avem+int(pingc*3)
        end = int(end)
    return start, end

def moving_average_max(a, n = pingc) :
    weights = np.repeat(1.0, n)/n
    alist = np.convolve(a, weights, 'valid')
    maxi = np.argmax(alist)
    ratio = np.amax(alist)/np.mean(alist)
    # maxa = 0
    # maxi = 0
    # for k in range(len(alist)):
    # 	ave = alist[k]
    # 	if ave > maxa:
    # 	    maxa = ave
    # 	    maxi = k
    return maxi, ratio

def find_second_window(starts_arr, ends_arr):
    # 0 overlap, get rid of current end and compare again
    # 1 overlap, and the other 2 also overlaps, use the first overlaps, else get rid of the nonoverlap
    # 1 overlap and the other 1 not, get rid fo the nonoverlap
    # 2 overlaps and the other 1 not, get rid of the nonoverlap
    # 3 overlaps, use all
    comp = np.ndarray(shape=(4, 4))
    for i in range(4):
        comp[i] = [starts_arr[i] < ends for ends in ends_arr]
    comp = np.triu(np.multiply(comp, np.transpose(comp)))
    true_index = np.where(comp)
    true_index_0 = []
    true_index_1 = []
    for i in range(len(true_index[0])):
        if true_index[0][i] != true_index[1][i]:
            true_index_0.append(true_index[0][i])
            true_index_1.append(true_index[1][i])
    true_index = np.array([true_index_0, true_index_1])
    true_index_f = list(true_index.flatten())
    count = [true_index_f.count(i) for i in range(4)]

    # use only 3 or 4 channel overlap
    if true_index.shape[1] == 6:
        return max(starts_arr), min(ends_arr), 4
    elif true_index.shape[1] < 3:
        return None, None, 0
    elif true_index.shape[1] == 3 and 0 not in count:
        return None, None, 0
    else:
        # if true_index.shape[1] == 3 and 0 in count:
        #     index = np.where(count == 0)
        # elif true_index.shape[1] == 4:
        #
        # elif true_index.shape[1] == 5:
        # this is throwing one channel out arbitrary, should be optimized to preserve the earlier one later
        index = np.argmin(count)
        starts_arr.pop(index)
        ends_arr.pop(index)
        return max(starts_arr), min(ends_arr), 3





def first_window(data1, data2, data3, data4):
    datasq1 = np.absolute(data1)
    datasq2 = np.absolute(data2)
    datasq3 = np.absolute(data3)
    datasq4 = np.absolute(data4)
    datasq = datasq1 + datasq2 + datasq3 + datasq4

    avem, ratio = moving_average_max(datasq)

    start, end = find_first_window(datasq, avem)
    print(start, end)


    dataw1 = datasq1[start:(end+1)]
    dataw2 = datasq2[start:(end+1)]
    dataw3 = datasq3[start:(end+1)]
    dataw4 = datasq4[start:(end+1)]

    return dataw1, dataw2, dataw3, dataw4, ratio

def second_window(dataw1, dataw2, dataw3, dataw4):
    starts1, ends1 = moving_average_increase(dataw1, False, spc*5)
    starts2, ends2 = moving_average_increase(dataw2, False, spc*5)
    starts3, ends3 = moving_average_increase(dataw3, False, spc*5)
    starts4, ends4 = moving_average_increase(dataw4, False, spc*5)
    starts, ends, overlap = find_second_window([starts1, starts2, starts3, starts4], [ends1, ends2, ends3, ends4])
    # starts = max([starts1, starts2, starts3, starts4])
    # ends = min([ends1, ends2, ends3, ends4])
    print([starts1, starts2, starts3, starts4], [ends1, ends2, ends3, ends4])
    print(starts, ends, overlap)

    if not overlap:
        return None, None, None, None, 0

    datasw1 = dataw1[starts:(ends+1)]
    datasw2 = dataw2[starts:(ends+1)]
    datasw3 = dataw3[starts:(ends+1)]
    datasw4 = dataw4[starts:(ends+1)]

    return datasw1, datasw2, datasw3, datasw4, overlap

def second_window_less(dataw1, dataw2, dataw3, dataw4):
    starts1, ends1 = moving_average_increase(dataw1, False, spc*5)
    starts2, ends2 = moving_average_increase(dataw2, False, spc*5)
    starts3, ends3 = moving_average_increase(dataw3, False, spc*5)
    starts4, ends4 = moving_average_increase(dataw4, False, spc*5)
    starts, ends, overlap = find_second_window([starts1, starts2, starts3, starts4], [ends1, ends2, ends3, ends4])
    print([starts1, starts2, starts3, starts4], [ends1, ends2, ends3, ends4])
    print(starts, ends, overlap)
    # starts = max([starts1, starts2, starts3, starts4])
    # ends = min([ends1, ends2, ends3, ends4])

    if not overlap:
        return None, None, None, None, 0

    datasw1 = dataw1[starts:(ends+1)]
    datasw2 = dataw2[starts:(ends+1)]
    datasw3 = dataw3[starts:(ends+1)]
    datasw4 = dataw4[starts:(ends+1)]

    return datasw1, datasw2, datasw3, datasw4, overlap

def second_window_less_less(dataw1, dataw2, dataw3, dataw4):
    starts1, ends1 = moving_average_increase(dataw1, True, spc*10)
    starts2, ends2 = moving_average_increase(dataw2, True, spc*10)
    starts3, ends3 = moving_average_increase(dataw3, True, spc*10)
    starts4, ends4 = moving_average_increase(dataw4, True, spc*10)
    starts, ends, overlap = find_second_window([starts1, starts2, starts3, starts4], [ends1, ends2, ends3, ends4])
    print([starts1, starts2, starts3, starts4], [ends1, ends2, ends3, ends4])
    print(starts, ends, overlap)
    # starts = max([starts1, starts2, starts3, starts4])
    # ends = min([ends1, ends2, ends3, ends4])

    if not overlap:
        return None, None, None, None, 0

    datasw1 = dataw1[starts:(ends+1)]
    datasw2 = dataw2[starts:(ends+1)]
    datasw3 = dataw3[starts:(ends+1)]
    datasw4 = dataw4[starts:(ends+1)]

    return datasw1, datasw2, datasw3, datasw4, overlap

def intp_window(datasw1, datasw2, datasw3, datasw4):
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

    intp_len = int(math.ceil(len(data1_intp)/4))
    data1_intpw = data1_intp[intp_len*2: intp_len*3]
    data2_intpw = data2_intp[intp_len*2: intp_len*3]
    data3_intpw = data3_intp[intp_len*2: intp_len*3]
    data4_intpw = data4_intp[intp_len*2: intp_len*3]

    return data1_intpw, data2_intpw, data3_intpw, data4_intpw

def xcrr(data1_intpw, data2_intpw, data3_intpw, data4_intpw):
    xcrr_12 = correlate(data1_intpw, data2_intpw, mode = 'full')
    xcrr_13 = correlate(data1_intpw, data3_intpw, mode = 'full')
    xcrr_34 = correlate(data3_intpw, data4_intpw, mode = 'full')

# when max_xcrr is left to the center, in1 is left to in2, in1 earlier than in2, negative p_diff
    diff_12 = (np.argmax(xcrr_12) - len(data1_intpw) + 1)/(10*fs)*vsound
    diff_13 = (np.argmax(xcrr_13) - len(data1_intpw) + 1)/(10*fs)*vsound
    diff_34 = (np.argmax(xcrr_34) - len(data1_intpw) + 1)/(10*fs)*vsound

    return diff_12, diff_13, diff_34


if __name__ == "__main__":

    hp1 = np.array([0, 0, 0])
    hp2 = np.array([0, -spac, 0])
    hp3 = np.array([-spac, 0, 0])
    hp4 = np.array([-spac, -spac, 0])


    if freq == 25000:
        bw = 1400
        k = 10
    elif freq == 30000:
        bw = 1400
        k = 20
    elif freq == 35000:
        bw = 2400
        k = 60
    # freq == 40000 and other cases
    else:
        bw = 1600
        k = 60

    filepath = sys.argv[1]

    df = pandas.read_csv(filepath, skiprows=[1], skipinitialspace=True)
    data1 = df["Channel 0"].tolist()
    data2 = df["Channel 1"].tolist()
    data3 = df["Channel 2"].tolist()
    data4 = df["Channel 3"].tolist()


    dataf1 = cheby2_bandpass_filter(data1, freq-bw/2, freq+bw/2, fs, k)
    dataf2 = cheby2_bandpass_filter(data2, freq-bw/2, freq+bw/2, fs, k)
    dataf3 = cheby2_bandpass_filter(data3, freq-bw/2, freq+bw/2, fs, k)
    dataf4 = cheby2_bandpass_filter(data4, freq-bw/2, freq+bw/2, fs, k)


    f_len = int(math.ceil(len(dataf1)/2))
    dataf1_1 = dataf1[13000:f_len]
    dataf1_2 = dataf1[f_len:]
    dataf2_1 = dataf2[13000:f_len]
    dataf2_2 = dataf2[f_len:]
    dataf3_1 = dataf3[13000:f_len]
    dataf3_2 = dataf3[f_len:]
    dataf4_1 = dataf4[13000:f_len]
    dataf4_2 = dataf4[f_len:]


    dataw1_1, dataw2_1, dataw3_1, dataw4_1, ratio_1 = first_window(dataf1_1, dataf2_1, dataf3_1, dataf4_1)
    dataw1_2, dataw2_2, dataw3_2, dataw4_2, ratio_2 = first_window(dataf1_2, dataf2_2, dataf3_2, dataf4_2)
    print(ratio_1, ratio_2)

    # plt.figure()
    # plt.plot(dataf1)
    # plt.plot(dataf2)
    # plt.plot(dataf3)
    # plt.plot(dataf4)
    # plt.legend(['1', '2', '3', '4'])
    # plt.show()

    if ratio_1 > 3:
        if ratio_1 > 10:
            datasw1_1, datasw2_1, datasw3_1, datasw4_1, overlap_1 = second_window(dataw1_1, dataw2_1, dataw3_1, dataw4_1)
        elif ratio_1 > 5:
            datasw1_1, datasw2_1, datasw3_1, datasw4_1, overlap_1 = second_window_less(dataw1_1, dataw2_1, dataw3_1, dataw4_1)
        else:
            datasw1_1, datasw2_1, datasw3_1, datasw4_1, overlap_1 = second_window_less_less(dataw1_1, dataw2_1, dataw3_1, dataw4_1)
        if overlap_1:
            data1_intpw_1, data2_intpw_1, data3_intpw_1, data4_intpw_1 = intp_window(datasw1_1, datasw2_1, datasw3_1, datasw4_1)
            diff_12_1, diff_13_1, diff_34_1 = xcrr(data1_intpw_1, data2_intpw_1, data3_intpw_1, data4_intpw_1)
            print("12", diff_12_1*(10*fs)/vsound, "13", diff_13_1*(10*fs)/vsound, "34", diff_34_1*(10*fs)/vsound)
            diff_12_1 = check_phase(diff_12_1)
            diff_13_1 = check_phase(diff_13_1)
            diff_34_1 = check_phase(diff_34_1)
            if not(diff_12_1 is None or diff_13_1 is None or diff_34_1 is None):
                data_val_1 = (hp1, hp2, hp3, hp4, diff_12_1, diff_13_1, diff_34_1)
                solution_1 = fsolve(system, (0, 0, -9), args=data_val_1)
                print("first", solution_1)
            else:
                print("eliminate data1, phase check")
        else:
            print("eliminate data1, overlap check")
    else:
        print("eliminate data1")

    if ratio_2 > 3:
        if ratio_2 > 10:
            datasw1_2, datasw2_2, datasw3_2, datasw4_2, overlap_2 = second_window(dataw1_2, dataw2_2, dataw3_2, dataw4_2)
        elif ratio_2 > 5:
            datasw1_2, datasw2_2, datasw3_2, datasw4_2, overlap_2 = second_window_less(dataw1_2, dataw2_2, dataw3_2, dataw4_2)
        else:
            datasw1_2, datasw2_2, datasw3_2, datasw4_2, overlap_2 = second_window_less_less(dataw1_2, dataw2_2, dataw3_2, dataw4_2)
        if overlap_2:
            data1_intpw_2, data2_intpw_2, data3_intpw_2, data4_intpw_2 = intp_window(datasw1_2, datasw2_2, datasw3_2, datasw4_2)
            diff_12_2, diff_13_2, diff_34_2 = xcrr(data1_intpw_2, data2_intpw_2, data3_intpw_2, data4_intpw_2)
            print("12", diff_12_2*(10*fs)/vsound, "13", diff_13_2*(10*fs)/vsound, "34", diff_34_2*(10*fs)/vsound)
            diff_12_2 = check_phase(diff_12_2)
            diff_13_2 = check_phase(diff_13_2)
            diff_34_2 = check_phase(diff_34_2)
            if not(diff_12_2 is None or diff_13_2 is None or diff_34_2 is None):
                data_val_2 = (hp1, hp2, hp3, hp4, diff_12_2, diff_13_2, diff_34_2)
                solution_2 = fsolve(system, (0, 0, -9), args=data_val_2)
                print("second", solution_2)
            else:
                print("eliminate data2, phase check")
        else:
            print("eliminate data2, overlap check")
    else:
        print("eliminate data2")
