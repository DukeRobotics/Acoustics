import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import cheby2, lfilter
from scipy.optimize import leastsq
import sys
import math

fs = 130000

ts = 3
pingc = fs*0.004
vsound = 1481
spac = 0.016
cycle = 1/float(fs)
bandpassw = 500
fftfreqw = 500
freq = 23500

def moving_average_double(a, n = pingc/10):
    weights = np.repeat(1.0, n)/n
    alist = np.convolve(a, weights, 'valid')
    lasta = alist[0]
    for k in range(len(alist)):
    	ave = alist[k]
    	if ave > lasta*5:
    	    return int(k+n)
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


def cheby2_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = cheby2(order, 5, [low, high], btype='bandpass')
    return b, a

def clean_phase(phase):
    if phase < -2*np.pi:
        return np.remainder(phase, -2*np.pi)
    if phase > 2*np.pi:
        return np.remainder(phase, 2*np.pi)
    return phase

def phase_diff(p1, p2):
    # p1 should be channel 0
    diff = p1 - p2
    if diff > np.pi:
        diff = diff - 2*np.pi
    if diff < -np.pi:
        diff = diff + 2*np.pi
    return diff


def cheby2_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = cheby2_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

if __name__ == "__main__":

    data0 = []
    data1 = []
    data2 = []
    with open("out.csv", 'rb') as filec:
        reader = csv.reader(filec)
        for row in reader:
            try:
                c1 = float(row[0])
                c2 = float(row[1])
                c3 = float(row[2])
		#print p
                data0.append(c1)
                data1.append(c2)
                data2.append(c3)
            except:
                continue
    try:
        out0 = cheby2_bandpass_filter(data0, freq-bandpassw/2, freq+bandpassw/2, fs)
        out1 = cheby2_bandpass_filter(data1, freq-bandpassw/2, freq+bandpassw/2, fs)
        out2 = cheby2_bandpass_filter(data2, freq-bandpassw/2, freq+bandpassw/2, fs)
    except Exception as e:
        print(e)

    #find window with moving_average
    out = out0+out1+out2
    outsq = np.absolute(out)
    avem = moving_average_max(outsq)
    start = 0
    end = len(out) - 1
    if avem-int(pingc*6) > start:
        start = avem-int(pingc*6)
        start = int(start)
    if avem+int(pingc) < end:
        end = avem+int(pingc)
        end = int(end)
    print "avem, start, end", avem, start, end
    outw0 = out0[start:(end+1)]
    outw1 = out1[start:(end+1)]
    outw2 = out2[start:(end+1)]
    outw = out[start:(end+1)]
    #dataw = data[start:(end+1)]
    time = np.linspace(start/fs, end/fs, end-start+1)

    aved = moving_average_double(np.absolute(outw))
    starts = aved
    ends = len(outw)-1
    # if aved-int(pingc/) > starts:
    #     starts = aved-int(pingc/50)
    #     starts = int(starts)
    if aved+int(pingc/30) < ends:
        ends = aved+int(pingc/30)
        ends = int(ends)
    print "aved, start, end", aved, starts, ends
    outsw0 = outw0[starts:(ends+1)]
    outsw1 = outw1[starts:(ends+1)]
    outsw2 = outw2[starts:(ends+1)]
    outsw = outw[starts:(ends+1)]


    guess_freq = 2*np.pi/fs*freq
    guess_phase = 0
    guess_std = 1
    t = np.arange(len(outsw0))

    optimize_func0 = lambda x: np.absolute(x[0])*t*np.sin(x[2]*(t+x[1])) - outsw0
    optimize_func1 = lambda x: np.absolute(x[0])*t*np.sin(x[2]*(t+x[1])) - outsw1
    optimize_func2 = lambda x: np.absolute(x[0])*t*np.sin(x[2]*(t+x[1])) - outsw2
    est_std0, est_phase0, est_freq0 = leastsq(optimize_func0, [guess_std, guess_phase, guess_freq])[0]
    est_std1, est_phase1, est_freq1 = leastsq(optimize_func1, [guess_std, guess_phase, guess_freq])[0]
    est_std2, est_phase2, est_freq2 = leastsq(optimize_func2, [guess_std, guess_phase, guess_freq])[0]
    data_fit0 = np.absolute(est_std0)*t*np.sin(est_freq0*(t+est_phase0))
    data_fit1 = np.absolute(est_std1)*t*np.sin(est_freq1*(t+est_phase1))
    data_fit2 = np.absolute(est_std2)*t*np.sin(est_freq2*(t+est_phase2))
    print "amp", est_std0, est_std1, est_std2
    print "freq", est_freq0, est_freq1, est_freq2
    print "phase", est_phase0, est_phase1, est_phase2


    # plt.figure()
    # plt.plot(out0)
    # plt.plot(out1)
    # plt.plot(out2)
    #
    # plt.figure()
    # plt.plot(outw0)
    # plt.plot(outw1)
    # plt.plot(outw2)
    # plt.plot()

    plt.figure()
    # plt.plot(outsw0)
    # plt.plot(outsw1)
    # plt.plot(outsw2)
    plt.plot(data_fit0)
    plt.plot(data_fit1)
    plt.plot(data_fit2)
    plt.show()


    # #fft
    # fft0 = np.fft.fft(outsw0)
    # ffta0 = np.absolute(fft0)
    # fft1 = np.fft.fft(outsw1)
    # ffta1 = np.absolute(fft1)
    # fft2 = np.fft.fft(outsw2)
    # ffta2 = np.absolute(fft2)
    # fft = np.fft.fft(outsw)
    # ffta = np.absolute(fft)
    # #result = 0
    # result = 1000000
    # resultc = 0
    # resultf = 0
    # resulti = 0
    #
    # #find max
    # timew = (end-start+1)/float(fs)
    # print timew
    # # for i in range(int(len(ffta)/2)):
    # #     f = i/timew;
    # #     if ffta[i] > result:
    # #         resultf = f
    # #         resulti = i
    # #         result = ffta[i]
	# #     resultc = i
    # for i in range(int(len(ffta)/2)):
    #     f = i/timew
    #     if np.absolute(f-freq) < result:
    #         resultf = f
    #         resulti = i
    #         result = np.absolute(f-freq)
	#     resultc = i
    # if np.absolute(resultf-freq)>fftfreqw:
    #     print "fft output wrong max magnitude for freq, bad data"
    #     print "resultf", resultf
    #     sys.exit()

    # resultp0 = np.angle(fft0[resultc]) #origin point
    # resultp1 = np.angle(fft1[resultc])
    # resultp2 = np.angle(fft2[resultc])
    resultp0 = clean_phase(est_phase0)
    resultp1 = clean_phase(est_phase1)
    resultp2 = clean_phase(est_phase2)
    print resultp0, resultp1, resultp2
    cycle = 1/float(fs)
    dphase_x = phase_diff(resultp1, resultp0)
    dphase_y = phase_diff(resultp2, resultp0)

    kx = vsound * dphase_x/ (spac * 2 * math.pi * freq);
    ky = vsound * dphase_y/ (spac * 2 * math.pi * freq);
    kz2 = 1 - kx*kx - ky*ky
    #print "max0, max1, max2", max0, max1, max2
    print "dphase_x, dphase_y", dphase_x, dphase_y
    print "kz2, kx, ky", kz2, kx, ky
    heading = np.arctan2(ky, kx)
    try:
        elevation = math.acos(math.sqrt(kz2))
    except:
        elevation = "Elevation out of range"

    print "heading, elevation", heading, elevation

    # plt.figure()
    # plt.plot(out0)
    # plt.plot(out1)
    # plt.plot(out2)
    # #
    # # plt.figure()
    # # plt.plot(outw0)
    # # plt.plot(outw1)
    # # plt.plot(outw2)
    #
    # plt.show()
