import pandas
# correlate(in1, in2) = k: in2 faster than in1 by k
from scipy.signal import correlate, cheby2, lfilter, freqz
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
vsound = 1511.5
#nipple distance between hydrophone
spac = 0.0115
#allowed phase diff
dphase = math.pi*2/(vsound/freq)*spac

fft_w_size = 125

bw = 1600*3

def get_pdiff(parr1, parr2, start, end):
    pdlist = np.subtract(parr2, parr1)
    var = variance_list(pdlist, int(pingc/fft_w_size/2))
    phase_start = moving_average_min(var[start:end])+start
    phase_end = phase_start+int(pingc/fft_w_size)
    pdiff = np.mean(correct_phase(pdlist[phase_start:phase_end]))
    print(phase_start)
    plt.figure()
    plt.plot(mlist1)
    plt.figure()
    plt.plot(pdlist)
    plt.show()
    return pdiff

def correct_phase(arr):
    result = []
    for phase in arr:
        if abs(phase) > np.pi:
            result.append(-phase/abs(phase)*(2*np.pi-abs(phase)))
        else:
            result.append(phase)
    return result

def fft_sw(xn, freq):
    n = np.array(range(len(xn)))
    exp = np.exp(-1j*freq/fs*2*np.pi*n)
    #print(xn, n)
    return np.dot(xn, exp)

def moving_average_max(a, n = 3*int(pingc/fft_w_size)) :
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

def moving_average_min(a, n = int(pingc/fft_w_size) - int(pingc/fft_w_size/2)) :
    weights = np.repeat(1.0, n)/n
    alist = np.convolve(a, weights, 'valid')
    mina = sys.maxsize
    mini = 0
    for k in range(len(alist)):
    	ave = alist[k]
    	if ave < mina:
    	    mina = ave
    	    mini = k
    return mini

def fft(xn, freq, w_size):
    p_list = []
    m_list = []
    for i in range(math.floor(len(xn)/w_size)):
        xn_s = xn[i*w_size:(i+1)*w_size]
        ft = fft_sw(xn_s, freq)
        # print(ft)
        phase = np.angle(ft)
        mag = np.absolute(ft)
        p_list.append(phase)
        m_list.append(mag)
    return p_list, m_list

def variance_list(arr, window):
    result = []
    for i in range(len(arr) - window+1):
        result.append(np.var(arr[i:i+window]))
    return result


def diff_equation(hp1, hp2, target, t_diff):
    return (np.linalg.norm(target-hp1) - np.linalg.norm(target-hp2)) - t_diff

def system(target, *data):
    hp1, hp2, hp3, hp4, diff_12, diff_13, diff_34= data
    return (diff_equation(hp1, hp2, target, diff_12), diff_equation(hp1, hp3, target, diff_13), diff_equation(hp3, hp4, target, diff_34))

if __name__ == "__main__":

    filepath = sys.argv[1]

    df = pandas.read_csv(filepath, skiprows=[1], skipinitialspace=True)
    print("running ", filepath)
    data1 = df["Channel 0"].tolist()
    data2 = df["Channel 1"].tolist()
    data3 = df["Channel 2"].tolist()
    data4 = df["Channel 3"].tolist()

    # plt.figure()
    # plt.plot(data1)
    # plt.plot(data2)
    # plt.plot(data3)
    # plt.plot(data4)
    # plt.legend(['1', '2', '3', '4'])
    # plt.show()

    # from data we know the range of valid stuff is 3ping, 3*0.004*625000/125 = 60
    plist1, mlist1 = fft(data1[int(len(data1)/2):], freq, fft_w_size)
    plist2, mlist2 = fft(data2[int(len(data1)/2):], freq, fft_w_size)
    plist3, mlist3 = fft(data3[int(len(data1)/2):], freq, fft_w_size)
    plist4, mlist4 = fft(data4[int(len(data1)/2):], freq, fft_w_size)
    maxi1 = moving_average_max(mlist1)
    maxi2= moving_average_max(mlist2)
    maxi3 = moving_average_max(mlist3)
    maxi4= moving_average_max(mlist4)
    mag_start = max([maxi1, maxi2, maxi3, maxi4])
    mag_end = min([maxi1, maxi2, maxi3, maxi4]) + 3*int(pingc/fft_w_size)

    # pdlist12 = np.subtract(plist2, plist1)
    # var12 = variance_list(pdlist12, int(pingc/fft_w_size/2))
    # phase_start12 = moving_average_min(var12[mag_start:mag_end])+mag_start
    # phase_end12 = phase_start12+int(pingc/fft_w_size)
    # pdiff12 = np.mean(correct_phase(pdlist12[phase_start12:phase_end12]))

    pdiff12 = get_pdiff(plist1, plist2, mag_start, mag_end)
    pdiff13 = get_pdiff(plist1, plist3, mag_start, mag_end)
    pdiff34 = get_pdiff(plist3, plist4, mag_start, mag_end)

    # pdlist13 = np.subtract(plist3, plist1)
    # var13 = variance_list(pdlist13, int(pingc/fft_w_size/2))
    # phase_start13 = moving_average_min(var13[mag_start:mag_end])+mag_start
    # phase_end13 = phase_start13+int(pingc/fft_w_size)
    # pdiff13 = np.mean(correct_phase(pdlist13[phase_start13:phase_end13]))
    #
    # pdlist34 = np.subtract(plist4, plist3)
    # var34 = variance_list(pdlist34, int(pingc/fft_w_size/2))
    # phase_start34 = moving_average_min(var34[mag_start:mag_end])+mag_start
    # phase_end34 = phase_start34+int(pingc/fft_w_size)
    # pdiff34 = np.mean(correct_phase(pdlist34[phase_start34:phase_end34]))
    # plist12, mlist12 = fft(data1, freq, 125)
    # plist22, mlist22 = fft(data2, freq, 125)
    # pdlist2 = np.subtract(plist12, plist22)
    # print(len(plist))

    # plt.figure()
    # #plt.plot(pdlist)
    # # plt.plot(pdlist12)
    # # plt.plot(pdlist13)
    # plt.plot(pdlist34)
    # # # plt.title("phase diff")
    # # plt.legend(['12', '13', '34'])
    # plt.figure()
    # plt.plot(mlist1)
    # # # plt.plot(mlist2)
    # # # plt.legend(['mag1', 'mag2'])
    # # # plt.figure()
    # # # plt.plot(var)
    # # # plt.title("variance")
    # # # plt.figure()
    # # # plt.plot(data1)
    # # # plt.legend(['filtered'])
    # plt.show()

    # diff_12 = (np.argmax(xcrr_12) - len(datasw1) + 1)
    # diff_23 = (np.argmax(xcrr_23) - len(datasw1) + 1)
    # diff_34 = (np.argmax(xcrr_34) - len(datasw1) + 1)
    print("pdiff_12", pdiff12, "pdiff_13", pdiff13, "pdiff_34", pdiff34)
    diff12 = pdiff12/2/np.pi*vsound/freq
    diff13 = pdiff13/2/np.pi*vsound/freq
    diff34 = pdiff34/2/np.pi*vsound/freq

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
    target = np.array([18.59, 27.54, -3.14])
    diff_12 = np.linalg.norm(target-hp1) - np.linalg.norm(target-hp2)
    diff_13 = np.linalg.norm(target-hp1) - np.linalg.norm(target-hp3)
    diff_34 = np.linalg.norm(target-hp3) - np.linalg.norm(target-hp4)
    print(diff_12/vsound*freq*2*np.pi, diff_13/vsound*freq*2*np.pi, diff_34/vsound*freq*2*np.pi)


    data_val = (hp1, hp2, hp3, hp4, diff12, diff13, diff34)
    guess = (18, 30, -3)
    solution = fsolve(system, guess, args=data_val) #18, 30.5, -3.05
    print("initial guess", guess)
    print("x, y, z", solution)
