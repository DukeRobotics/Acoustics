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
freq = 30000
#speed of sound in water
vsound = 1481
#nipple distance between hydrophone
spac = 0.012
#allowed phase diff
dphase = math.pi*2/(vsound/freq)*spac

#bw = 1400

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

def first_window(data1, data2, data3, data4):
    datasq1 = np.absolute(data1)
    datasq2 = np.absolute(data2)
    datasq3 = np.absolute(data3)
    datasq4 = np.absolute(data4)
    datasq = datasq1 + datasq2 + datasq3 + datasq4

    avem = moving_average_max(datasq)

    start, end = find_first_window(datasq, avem)
    print(start+13000, end+13000)


    # # plt.figure()
    # # plt.plot(data1[start:end])
    # # plt.plot(datasq1[start:(end+1)])
    # # stft has frequency on the y axis and time on the x axis
    # f, t, Zxx = signal.stft(data1[start:end], fs)
    # # index = np.where(f==3000)
    # # print(index)
    # plt.figure()
    # plt.plot(t, np.absolute(Zxx[13]))
    # plt.figure()
    # plt.plot(t, np.angle(Zxx[13]))
    # # plt.show()
    # # print(f)
    # # plt.figure()
    # # plt.pcolormesh(t, f, np.angle(Zxx))
    # # plt.title('STFT Magnitude')
    # # plt.ylabel('Frequency [Hz]')
    # # plt.xlabel('Time [sec]')
    # plt.show()

    dataw1 = datasq1[start:(end+1)]
    dataw2 = datasq2[start:(end+1)]
    dataw3 = datasq3[start:(end+1)]
    dataw4 = datasq4[start:(end+1)]

    return dataw1, dataw2, dataw3, dataw4

def second_window(dataw1, dataw2, dataw3, dataw4):
    starts1, ends1 = moving_average_increase(dataw1)
    starts2, ends2 = moving_average_increase(dataw2)
    starts3, ends3 = moving_average_increase(dataw3)
    starts4, ends4 = moving_average_increase(dataw4)
    starts = max([starts1, starts2, starts3, starts4])
    ends = min([ends1, ends2, ends3, ends4])

    datasw1 = dataw1[starts:(ends+1)]
    datasw2 = dataw2[starts:(ends+1)]
    datasw3 = dataw3[starts:(ends+1)]
    datasw4 = dataw4[starts:(ends+1)]

    return datasw1, datasw2, datasw3, datasw4

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

    intp_len = math.ceil(len(data1_intp)/4)
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

    # # plt.figure()
    # # plt.plot(data1[start:end])
    # # plt.plot(datasq1[start:(end+1)])
    # # stft has frequency on the y axis and time on the x axis
    # f, t, Zxx = signal.stft(data1[582668:597668], fs)
    # # index = np.where(f==3000)
    # # print(index)
    # plt.figure()
    # plt.plot(t, np.absolute(Zxx[13]))
    # plt.figure()
    # plt.plot(t, np.angle(Zxx[13]))
    # # plt.show()
    # # print(f)
    # # plt.figure()
    # # plt.pcolormesh(t, f, np.angle(Zxx))
    # # plt.title('STFT Magnitude')
    # # plt.ylabel('Frequency [Hz]')
    # # plt.xlabel('Time [sec]')
    # plt.show()

    dataf1 = cheby2_bandpass_filter(data1, freq-bw/2, freq+bw/2, fs, k)
    dataf2 = cheby2_bandpass_filter(data2, freq-bw/2, freq+bw/2, fs, k)
    dataf3 = cheby2_bandpass_filter(data3, freq-bw/2, freq+bw/2, fs, k)
    dataf4 = cheby2_bandpass_filter(data4, freq-bw/2, freq+bw/2, fs, k)

    plt.figure()
    plt.plot(dataf1)
    plt.plot(dataf2)
    plt.plot(dataf3)
    plt.plot(dataf4)
    plt.legend(['1', '2', '3', '4'])
    plt.show()

    f_len = math.ceil(len(dataf1)/2)
    dataf1_1 = dataf1[13000:f_len]
    dataf1_2 = dataf1[f_len:]
    dataf2_1 = dataf2[13000:f_len]
    dataf2_2 = dataf2[f_len:]
    dataf3_1 = dataf3[13000:f_len]
    dataf3_2 = dataf3[f_len:]
    dataf4_1 = dataf4[13000:f_len]
    dataf4_2 = dataf4[f_len:]

    # plt.figure()
    # plt.plot(dataf1_1)
    # plt.plot(dataf2_1)
    # plt.plot(dataf3_1)
    # plt.plot(dataf4_1)
    # plt.legend(['1', '2', '3', '4'])
    # plt.figure()
    # plt.plot(dataf1_2)
    # plt.plot(dataf2_2)
    # plt.plot(dataf3_2)
    # plt.plot(dataf4_2)
    # plt.legend(['1', '2', '3', '4'])
    # plt.show()



    # datasq1_1 = np.absolute(dataf1_1[13000:])
    # datasq2_1 = np.absolute(dataf2_1[13000:])
    # datasq3_1 = np.absolute(dataf3_1[13000:])
    # datasq4_1 = np.absolute(dataf4_1[13000:])
    # datasq1_2 = np.absolute(dataf1_2)
    # datasq2_2 = np.absolute(dataf2_2)
    # datasq3_2 = np.absolute(dataf3_2)
    # datasq4_2 = np.absolute(dataf4_2)
    #since this is a rough window so we do only one max moving ave on the sum with length of ping
    # datasq_1 = datasq1_1+datasq2_1+datasq3_1+datasq4_1
    # datasq_2 = datasq1_2+datasq2_2+datasq3_2+datasq4_2
    #
    # avem_1 = moving_average_max(datasq_1)
    # avem_2 = moving_average_max(datasq_2)
    #
    # start_1, end_1 = find_first_window(datasq_1, avem_1)
    # start_2, end_2 = find_first_window(datasq_2, avem_2)
    #
    #
    # dataw1_1 = datasq1_1[start_1:(end_1+1)]
    # dataw2_1 = datasq2_1[start_1:(end_1+1)]
    # dataw3_1 = datasq3_1[start_1:(end_1+1)]
    # dataw4_1 = datasq4_1[start_1:(end_1+1)]
    #
    # dataw1_2 = datasq1_2[start_2:(end_2+1)]
    # dataw2_2 = datasq2_2[start_2:(end_2+1)]
    # dataw3_2 = datasq3_2[start_2:(end_2+1)]
    # dataw4_2 = datasq4_2[start_2:(end_2+1)]

    dataw1_1, dataw2_1, dataw3_1, dataw4_1 = first_window(dataf1_1, dataf2_1, dataf3_1, dataf4_1)
    dataw1_2, dataw2_2, dataw3_2, dataw4_2 = first_window(dataf1_2, dataf2_2, dataf3_2, dataf4_2)
    print('length of dataw1 ', len(dataw1_1))

    # plt.figure()
    # plt.plot(dataw1_1)
    # plt.plot(dataw2_1)
    # plt.plot(dataw3_1)
    # plt.plot(dataw4_1)
    # plt.show()

    # f, t, Zxx = signal.stft(dataw1_1, fs)
    # plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=3.5)
    # plt.title('STFT Magnitude')
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.show()

    # t = np.arange(len(dataw1_1))
    #
    # sp = np.fft.fft(dataw1_1)
    # freq = np.fft.fftfreq(t.shape[-1])*fs
    # plt.figure()
    # plt.plot(freq, np.absolute(sp))
    # plt.figure()
    # plt.plot(freq, np.angle(sp))
    # plt.show()

    # plt.figure()
    # plt.plot(dataw1_1)
    # plt.plot(dataw2_1)
    # plt.plot(dataw3_1)
    # plt.plot(dataw4_1)
    # plt.legend(['2', '4'])
    # plt.figure()
    # plt.plot(dataw2_1)
    # # plt.plot(dataw2_2)
    # # plt.plot(dataw3_2)
    # # plt.plot(dataw4_2)
    # # plt.legend(['1', '2', '3', '4'])
    # plt.show()


    # starts1_1, ends1_1 = moving_average_increase(dataw1_1)
    # starts2_1, ends2_1 = moving_average_increase(dataw2_1)
    # starts3_1, ends3_1 = moving_average_increase(dataw3_1)
    # starts4_1, ends4_1 = moving_average_increase(dataw4_1)
    # starts_1 = max([starts1_1, starts2_1, starts3_1, starts4_1])
    # ends_1 = min([ends1_1, ends2_1, ends3_1, ends4_1])
    #
    # starts1_2, ends1_2 = moving_average_increase(dataw1_2)
    # starts2_2, ends2_2 = moving_average_increase(dataw2_2)
    # starts3_2, ends3_2 = moving_average_increase(dataw3_2)
    # starts4_2, ends4_2 = moving_average_increase(dataw4_2)
    # starts_2 = max([starts1_2, starts2_2, starts3_2, starts4_2])
    # ends_2 = min([ends1_2, ends2_2, ends3_2, ends4_2])
    # # print(starts, ends)
    #
    #
    # # #use moving_average_double to locate the very first part
    # # aved1 = moving_average_increase(dataw1)
    # # aved2 = moving_average_increase(dataw2)
    # # aved3 = moving_average_increase(dataw3)
    # # aved4 = moving_average_increase(dataw4)
    # # aved = max([aved1, aved2, aved3, aved4])
    # # print(aved1, aved2, aved3, aved4)
    # #
    # # starts, ends = find_second_window(dataw, aved)
    # # print(start+starts+13000, start+ends+13000)
    #
    # # plt.figure()
    # # plt.plot(np.absolute(dataw))
    # plt.figure()
    # plt.plot(dataw1)
    # plt.plot(dataw2)
    # plt.plot(dataw3)
    # plt.plot(dataw4)
    # plt.legend(['1', '2', '3', '4'])
    # plt.show()
    #
    #
    # # # print "aved, start, end", aved, starts, ends
    # datasw1 = dataw1[starts:(ends+1)]
    # datasw2 = dataw2[starts:(ends+1)]
    # datasw3 = dataw3[starts:(ends+1)]
    # datasw4 = dataw4[starts:(ends+1)]

    datasw1_1, datasw2_1, datasw3_1, datasw4_1 = second_window(dataw1_1, dataw2_1, dataw3_1, dataw4_1)
    datasw1_2, datasw2_2, datasw3_2, datasw4_2 = second_window(dataw1_2, dataw2_2, dataw3_2, dataw4_2)

# not sure if we really need the interpolation

    # 10 times sampling points, so 10 times sampling frequency
#     time = np.linspace(0, len(datasw1)-1 ,len(datasw1)*10 - 9)
#
#     data1f = interp1d(range(0, len(datasw1)), datasw1, kind = 'cubic')
#     data2f = interp1d(range(0, len(datasw2)), datasw2, kind = 'cubic')
#     data3f = interp1d(range(0, len(datasw3)), datasw3, kind = 'cubic')
#     data4f = interp1d(range(0, len(datasw4)), datasw4, kind = 'cubic')
#
#     data1_intp = data1f(time)
#     data2_intp = data2f(time)
#     data3_intp = data3f(time)
#     data4_intp = data4f(time)
#
# # only use the middle section to avoid distortion
#
#     intp_len = math.ceil(len(data1_intp)/4)
#     data1_intpw = data1_intp[intp_len*2: intp_len*3]
#     data2_intpw = data2_intp[intp_len*2: intp_len*3]
#     data3_intpw = data3_intp[intp_len*2: intp_len*3]
#     data4_intpw = data4_intp[intp_len*2: intp_len*3]

    data1_intpw_1, data2_intpw_1, data3_intpw_1, data4_intpw_1 = intp_window(datasw1_1, datasw2_1, datasw3_1, datasw4_1)
    data1_intpw_2, data2_intpw_2, data3_intpw_2, data4_intpw_2 = intp_window(datasw1_2, datasw2_2, datasw3_2, datasw4_2)

# # cross correlation
#
#     xcrr_12 = correlate(data1_intpw, data2_intpw, mode = 'full')
#     xcrr_23 = correlate(data2_intpw, data3_intpw, mode = 'full')
#     xcrr_34 = correlate(data3_intpw, data4_intpw, mode = 'full')
#
# # when max_xcrr is left to the center, in1 is left to in2, in1 earlier than in2, negative p_diff
#     diff_12 = (np.argmax(xcrr_12) - len(data1_intpw) + 1)/(10*fs)*vsound
#     diff_23 = (np.argmax(xcrr_23) - len(data1_intpw) + 1)/(10*fs)*vsound
#     diff_34 = (np.argmax(xcrr_34) - len(data1_intpw) + 1)/(10*fs)*vsound
    diff_12_1, diff_13_1, diff_34_1 = xcrr(data1_intpw_1, data2_intpw_1, data3_intpw_1, data4_intpw_1)
    diff_12_2, diff_13_2, diff_34_2 = xcrr(data1_intpw_2, data2_intpw_2, data3_intpw_2, data4_intpw_2)
    print(diff_12_1*(10*fs)/vsound, diff_13_1*(10*fs)/vsound, diff_34_1*(10*fs)/vsound)
    print(diff_12_2*(10*fs)/vsound, diff_13_2*(10*fs)/vsound, diff_34_2*(10*fs)/vsound)

    # diff_12_2 = 27/(10*fs)*vsound
    # diff_23_2 = -62/(10*fs)*vsound
    # diff_34_2 = 2/(10*fs)*vsound

    # plt.figure()
    # plt.plot(dataf1_1)
    # plt.plot(dataf2_1)
    # plt.plot(dataf3_1)
    # plt.plot(dataf4_1)
    # plt.legend(['1', '2', '3', '4'])
    # plt.figure()
    # plt.plot(dataf1_2)
    # plt.plot(dataf2_2)
    # plt.plot(dataf3_2)
    # plt.plot(dataf4_2)
    # plt.legend(['1', '2', '3', '4'])
    # plt.show()


    plt.figure()
    plt.plot(data1_intpw_1)
    plt.plot(data2_intpw_1)
    plt.plot(data3_intpw_1)
    plt.plot(data4_intpw_1)
    plt.legend(['1', '2', '3', '4'])
    plt.figure()
    plt.plot(data1_intpw_2)
    plt.plot(data2_intpw_2)
    plt.plot(data3_intpw_2)
    plt.plot(data4_intpw_2)
    plt.legend(['1', '2', '3', '4'])
    plt.show()

    hp1 = np.array([0, 0, 0])
    hp2 = np.array([0, -spac, 0])
    hp3 = np.array([-spac, 0, 0])
    hp4 = np.array([-spac, -spac, 0])
    # target = np.array([-1, -1, -1])
    # diff_12 = np.linalg.norm(target-hp1) - np.linalg.norm(target-hp2)
    # diff_23 = np.linalg.norm(target-hp2) - np.linalg.norm(target-hp3)
    # diff_34 = np.linalg.norm(target-hp3) - np.linalg.norm(target-hp4)


    data_val_1 = (hp1, hp2, hp3, hp4, diff_12_1, diff_13_1, diff_34_1)
    solution_1 = fsolve(system, (0, 0, 0), args=data_val_1)
    print("first", solution_1)
    data_val_2 = (hp1, hp2, hp3, hp4, diff_12_2, diff_13_2, diff_34_2)
    solution_2 = fsolve(system, (0, 0, 0), args=data_val_2)
    print("second", solution_2)
