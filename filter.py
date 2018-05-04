import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import cheby2, lfilter
from scipy.optimize import leastsq
import sys
import math
import subprocess

fs = 130000

ts = 3
pingc = fs*0.004
vsound = 1481
spac = 0.01275
cycle = 1/float(fs)
bandpassw = 500
fftfreqw = 500


#running average get time section, fft get phase comparison, multichannels

def cheby2_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = cheby2(order, 5, [low, high], btype='bandpass')
    return b, a

def cheby2_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = cheby2_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def phase_diff(p1, p2):
    # p2 should be channel 0
    diff = p1 - p2
    if diff > np.pi:
        diff = diff - 2*np.pi
    if diff < -np.pi:
        diff = diff + 2*np.pi
    return diff

def clean_phase(phase):
    if phase < -2*np.pi:
        return np.remainder(phase, -2*np.pi)
    if phase > 2*np.pi:
        return np.remainder(phase, 2*np.pi)
    return phase

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

def sin_fit(guess_freq, outsw0, outsw1, outsw2):
    guess_std = 1
    guess_phase = 0
    t = np.arange(len(outsw0))

    optimize_func0 = lambda x: np.absolute(x[0])*t*np.sin(x[2]*t+x[1]) - outsw0
    est_std0, est_phase0, est_freq0 = leastsq(optimize_func0, [guess_std, guess_phase, guess_freq])[0]
    optimize_func1 = lambda x: np.absolute(x[0])*t*np.sin(est_freq0*t+x[1]) - outsw1
    optimize_func2 = lambda x: np.absolute(x[0])*t*np.sin(est_freq0*t+x[1]) - outsw2
    est_std1, est_phase1 = leastsq(optimize_func1, [guess_std, guess_phase])[0]
    est_std2, est_phase2 = leastsq(optimize_func2, [guess_std, guess_phase])[0]
    # data_fit0 = np.absolute(est_std0)*t*np.sin(est_freq0*t+est_phase0)
    # data_fit1 = np.absolute(est_std1)*t*np.sin(est_freq0*t+est_phase1)
    # data_fit2 = np.absolute(est_std2)*t*np.sin(est_freq0*t+est_phase2)
    return est_phase0, est_phase1, est_phase2

if __name__ == "__main__":
    #sampling
    data0 = []
    data1 = []
    data2 = []
    freq = int(sys.argv[1])
    process = subprocess.Popen(["/home/estellehe/Desktop/Linux_Drivers/USB/mcc-libusb/sampling", str(ts), str(fs)], stdout = subprocess.PIPE)
    stddata, stderror = process.communicate()
    if "fail" in stddata:
        print stddata
        sys.exit()

    #parse data from stdin
    datas = stddata.split("\n")
    for d in datas:
	ds = d.split(",")
        try:
            data0.append(float(ds[1]))
            data1.append(float(ds[2]))
            data2.append(float(ds[0]))
        except:
            continue
    data0 = data0[13000:]
    data1 = data1[13000:]
    data2 = data2[13000:]

    # with open("data.csv", 'rb') as filec:
    #     reader = csv.reader(filec)
    #     for row in reader:
    #         try:
    #             p = float(row[0])
	# 	#print p
    #             data.append(p)
    #         except:
    #             continue

    #bandpass
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
    mag = np.sum(np.absolute(out))
    print "magnitude", mag

    aved = moving_average_double(np.absolute(outw))
    starts = 0
    ends = len(outw)-1
    if aved-int(pingc/60) > starts:
        starts = aved-int(pingc/60)
        starts = int(starts)
    if aved+int(pingc/20) < ends:
        ends = aved+int(pingc/20)
        ends = int(ends)
    print "aved, start, end", aved, starts, ends
    outsw0 = outw0[starts:(ends+1)]
    outsw1 = outw1[starts:(ends+1)]
    outsw2 = outw2[starts:(ends+1)]
    outsw = outw[starts:(ends+1)]

    guess_freq = 2*np.pi/fs*freq

    est_phase0, est_phase1, est_phase2 = sin_fit(guess_freq, outsw0, outsw1, outsw2)



    #todo: the order of wave hitting hydrophone
    # max0 = moving_average(outw0, pingc)
    # max1 = moving_average(outw1, pingc)
    # max2 = moving_average(outw2, pingc)
    # order = [np.argmin([max0, max1, max2]), np.argmax([max0, max1, max2])]
    # order = [1, 2]

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
    # timew = (ends-starts+1)/float(fs)
    # print timew
    # for i in range(int(len(ffta)/2)):
    #     f = i/timew;
    #     if ffta[i] > result:
    #         resultf = f
    #         resulti = i
    #         result = ffta[i]
	#     resultc = i
    # for i in range(int(len(ffta)/2)):
    #     f = i/timew
    #     if np.absolute(f-freq) < result:
    #         resultf = f
    #         resulti = i
    #         result = np.absolute(f-freq)
	#     resultc = i
    #     #print fft[i], ffta[i], f
    # if np.absolute(resultf-freq)>fftfreqw:
    #     print "fft output wrong max magnitude for freq, bad data"
    #     print "resultf", resultf
    #     sys.exit()

    resultp0 = clean_phase(est_phase0) - np.pi*2/3
    resultp1 = clean_phase(est_phase1)
    resultp2 = clean_phase(est_phase2) + np.pi/2
    cycle = 1/float(fs)
    dphase_x = phase_diff(resultp1, resultp0)
    dphase_y = phase_diff(resultp2, resultp0)
    # if np.absolute(dphase_x) < 0.5 or np.absolute(dphase_x) > (np.pi-0.5) or np.absolute(dphase_y) < 0.5 or np.absolute(dphase_y) > (np.pi-0.5):
    #     starts = 0
    #     ends = len(outw)-1
    #     if aved-int(pingc/60) > starts:
    #         starts = aved-int(pingc/60)
    #         starts = int(starts)
    #     if aved+int(pingc/40) < ends:
    #         ends = aved+int(pingc/40)
    #         ends = int(ends)
    #     print "aved, start, end", aved, starts, ends
    #     outsw0 = outw0[starts:(ends+1)]
    #     outsw1 = outw1[starts:(ends+1)]
    #     outsw2 = outw2[starts:(ends+1)]
    #     outsw = outw[starts:(ends+1)]
    #     est_phase0, est_phase1, est_phase2 = sin_fit(guess_freq, outsw0, outsw1, outsw2)
    #     resultp0 = clean_phase(est_phase0)
    #     resultp1 = clean_phase(est_phase1)
    #     resultp2 = clean_phase(est_phase2)
    #     cycle = 1/float(fs)
    #     dphase_x = phase_diff(resultp1, resultp0)
    #     dphase_y = phase_diff(resultp2, resultp0)
    print "phase", est_phase0, est_phase1, est_phase2


    # if order[0] == 0:
    #     dphase_x = phase_diff(resultp1, resultp0)
    #     dphase_y = phase_diff(resultp2, resultp0)
    # elif order[1] == 0:
    #     dphase_x = -phase_diff(resultp0, resultp1) #should be phase(0) - phase(1)
    #     dphase_y = -phase_diff(resultp0, resultp2) #should be phase(0) - phase(2)
    # elif order[0] == 1:
    #     dphase_x = -phase_diff(resultp0, resultp1)
    #     dphase_y = phase_diff(resultp2, resultp0)
    # else:
    #     dphase_x = phase_diff(resultp1, resultp0)
    #     dphase_y = -phase_diff(resultp0, resultp2)
    # dphase_x = abs(resultp1 - resultp0) #01 as x direction
    # dphase_y = abs(resultp2 - resultp0) #02 as y direction
    #timediff = phase/(2pi*freq)

    #the nipple distance need to be half wavelength of the target frequency
    #since we can only get phase diff within one cycle, and half cycle gives us the order of channel
    # dphase_x = abs(resultp1 - resultp0) #0-1 as x direction
    # dphase_y = abs(resultp2 - resultp0) #0-2 as y direction
    # the horizontal angle is counterclock from positive x-axis(0-1)
    # the elevation angle is looking down from the bot (parallel would be 0 degree)
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
    print "resultp0, resultp1, resultp2, cycle", resultp0, resultp1, resultp2, cycle
    #print out[0]
    with open("out.csv", 'wb') as write:
        writer = csv.writer(write)
        for k in range(len(out)):
            writer.writerow([round(out0[k], 4), round(out1[k], 4), round(out2[k], 4), round(out[k], 4)])
    #plt.plot(out)
    plt.figure()
    plt.plot(outw0)
    plt.plot(outw1)
    plt.plot(outw2)

    plt.figure()
    plt.plot(outsw0)
    plt.plot(outsw1)
    plt.plot(outsw2)

    # plt.figure()
    # plt.plot(out)
    plt.show()
    # print len(data)
    #print out
    # subprocess.call(["rm", "testcsv"])
    # subprocess.call(["gcc", "-o", "testcsv", "testcsv.c", "-lfftw3", "-lm"])
