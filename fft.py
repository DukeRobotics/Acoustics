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
from mpl_toolkits import mplot3d

#sampling frequency
fs = 625000
#sampling time period
ts = 3
#number of sample taken during the ping
pingc = fs*0.004
#target frequency
freq = 40000
#speed of sound in water
vsound = 1511.5
#nipple distance between hydrophone
spac = 0.0115
#allowed phase diff
dphase = math.pi*2/(vsound/freq)*spac

fft_w_size = 125

bw = 1600*3

def get_pdiff(parr1, parr2, start, end):
    pdlist = correct_phase(np.subtract(parr2, parr1))
    var = variance_list(pdlist, int(pingc/fft_w_size/2))
    phase_start = moving_average_min(var[start:end])+start
    # check if lowest variance align with max mag interval, if not then bad data
    if abs(phase_start - start) > 5:
        return None
    phase_end = phase_start+int(pingc/fft_w_size)
    pdiff = np.mean(correct_phase(pdlist[phase_start:phase_end]))
    # print("phase start", phase_start)
    # print("phase end", phase_end)
    # plt.figure()
    # plt.plot(pdlist)
    # plt.title("phase diff")
    # plt.show()
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

def moving_average_max(a, n = int(pingc/fft_w_size)) :
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
    hp1, hp2, hp3, hp4, diff_12, diff_13, diff_34 = data
    return (diff_equation(hp1, hp2, target, diff_12), diff_equation(hp1, hp3, target, diff_13), diff_equation(hp3, hp4, target, diff_34))

def solver(guess, diff12, diff13, diff34):
    hp1 = np.array([0, 0, 0])
    hp2 = np.array([0, -spac, 0])
    hp3 = np.array([-spac, 0, 0])
    hp4 = np.array([-spac, -spac, 0])
    data_val = (hp1, hp2, hp3, hp4, diff12, diff13, diff34)
    solution = fsolve(system, guess, args=data_val)
    return solution

def data_to_pdiff(data1, data2, data3, data4):
    plist1, mlist1 = fft(data1, freq, fft_w_size)
    plist2, mlist2 = fft(data2, freq, fft_w_size)
    plist3, mlist3 = fft(data3, freq, fft_w_size)
    plist4, mlist4 = fft(data4, freq, fft_w_size)
    # plt.figure()
    # plt.plot(mlist1)
    # plt.show()

    # maxi1 = moving_average_max(mlist1)
    # maxi2= moving_average_max(mlist2)
    # maxi3 = moving_average_max(mlist3)
    # maxi4= moving_average_max(mlist4)
    mlist = np.add(np.add(mlist1, mlist2), np.add(mlist3, mlist4))
    maxi = moving_average_max(mlist)
    mag_start = maxi
    mag_end = maxi + int(pingc/fft_w_size)
    # mag_start = min([maxi1, maxi2, maxi3, maxi4])
    # mag_end = min([maxi1, maxi2, maxi3, maxi4]) + 3*int(pingc/fft_w_size)

    pdiff12 = get_pdiff(plist1, plist2, mag_start, mag_end)
    pdiff13 = get_pdiff(plist1, plist3, mag_start, mag_end)
    pdiff34 = get_pdiff(plist3, plist4, mag_start, mag_end)

    return pdiff12, pdiff13, pdiff34

def read_data(filepath):
    df = pandas.read_csv(filepath, skiprows=[1], skipinitialspace=True)
    print("running ", filepath)
    data1 = df["Channel 0"].tolist()
    data2 = df["Channel 1"].tolist()
    data3 = df["Channel 2"].tolist()
    data4 = df["Channel 3"].tolist()

    return data1, data2, data3, data4

def plot_3d(ccwh, downv):
    if abs(ccwh) < np.pi/2:
        xline = np.linspace(0, 10, 100)
    else :
        xline = np.linspace(-10, 0, 100)
    yline = xline*np.tan(ccwh)
    zline = -xline/np.cos(ccwh)*np.tan(downv)
    ax.plot3D(xline, yline, zline, 'gray')


if __name__ == "__main__":

    filepath = sys.argv[1]

    # df = pandas.read_csv(filepath, skiprows=[1], skipinitialspace=True)
    # # print("running ", filepath)
    # data1 = df["Channel 0"].tolist()
    # data2 = df["Channel 1"].tolist()
    # data3 = df["Channel 2"].tolist()
    # data4 = df["Channel 3"].tolist()

    ccwha = []
    downva = []
    for i in range(4):
        # data1, data2, data3, data4 = read_data(filepath.replace(".csv", str(i+1)+".csv"))
        data1, data2, data3, data4 = read_data(filepath.replace(".csv", "("+str(i+1)+").csv"))
        # data1, data2, data3, data4 = read_data(filepath)
        data1a = [data1[0:int(len(data1)/2)], data1[int(len(data1)/2):len(data1)]]
        data2a = [data2[0:int(len(data1)/2)], data2[int(len(data1)/2):len(data1)]]
        data3a = [data3[0:int(len(data1)/2)], data3[int(len(data1)/2):len(data1)]]
        data4a = [data4[0:int(len(data1)/2)], data4[int(len(data1)/2):len(data1)]]
        # plt.figure()
        # plt.plot(data1)
        # plt.plot(data1a[1])
        # plt.show()
        for j in range(2):
            data1 = data1a[j]
            data2 = data2a[j]
            data3 = data3a[j]
            data4 = data4a[j]
            pdiff12, pdiff13, pdiff34 = data_to_pdiff(data1, data2, data3, data4)
            print("\npdiff_12", pdiff12, "pdiff_13", pdiff13, "pdiff_34", pdiff34, "\n")

            if (pdiff12 is not None) and (pdiff13 is not None) and (pdiff34 is not None):
                diff12 = pdiff12/2/np.pi*vsound/freq
                diff13 = pdiff13/2/np.pi*vsound/freq
                diff34 = pdiff34/2/np.pi*vsound/freq

                guess = (3, -3, -4)
                # guess = (-1, 1, -3)
                # solution = solver(guess, diff12, diff13, diff34)
                x, y, z = solver(guess, diff12, diff13, diff34)
                print("initial guess", guess)
                print("x, y, z", x, y, z)
                ccwh = np.arctan2(y, x)
                ccwha.append(ccwh)
                print("horizontal angle", ccwh/np.pi*180)
                downv = np.arctan2(-z, np.sqrt(x**2 + y**2))
                downva.append(downv)
                print("vertical downward angle", downv/np.pi*180, "\n")

    plt.figure()
    ax = plt.axes(projection='3d')
    # ax.scatter3D([-1], [1], [-3]);
    ax.scatter3D([3], [-3], [-4]);
    for i in range(len(ccwha)):
        plot_3d(ccwha[i], downva[i])
    plt.show()

    # for checking pdiff across range data
    # pdiff12a = []
    # pdiff13a = []
    # pdiff34a = []
    # for i in range(4):
    #     data1, data2, data3, data4 = read_data(filepath.replace(".csv", "("+str(i+1)+").csv"))
    #     data11 = data1[0:int(len(data1)/2)]
    #     data21 = data2[0:int(len(data1)/2)]
    #     data31 = data3[0:int(len(data1)/2)]
    #     data41 = data4[0:int(len(data1)/2)]
    #     data12 = data1[int(len(data1)/2):len(data1)]
    #     data22 = data2[int(len(data1)/2):len(data1)]
    #     data32 = data3[int(len(data1)/2):len(data1)]
    #     data42 = data4[int(len(data1)/2):len(data1)]
    #     pdiff121, pdiff131, pdiff341 = data_to_pdiff(data11, data21, data31, data41)
    #     pdiff122, pdiff132, pdiff342 = data_to_pdiff(data12, data22, data32, data42)
    #     pdiff12a.append(pdiff121)
    #     pdiff12a.append(pdiff122)
    #     pdiff13a.append(pdiff131)
    #     pdiff13a.append(pdiff132)
    #     pdiff34a.append(pdiff341)
    #     pdiff34a.append(pdiff342)
    #
    # plt.figure()
    # plt.plot(pdiff12a)
    # plt.plot(pdiff13a)
    # plt.plot(pdiff34a)
    # plt.legend(['12', '13', '34'])
    # plt.show()
    #
    # pdiff12 = np.mean([x for x in pdiff12a if x is not None])
    # pdiff13 = np.mean([x for x in pdiff13a if x is not None])
    # pdiff34 = np.mean([x for x in pdiff34a if x is not None])

    # plt.figure()
    # plt.plot(data1)
    # plt.plot(data2)
    # plt.plot(data3)
    # plt.plot(data4)
    # plt.legend(['1', '2', '3', '4'])
    # plt.show()

    # from data we know the range of valid stuff is 3ping, 3*0.004*625000/125 = 60
    # plist1, mlist1 = fft(data1[0:int(len(data1)/2)], freq, fft_w_size)
    # plist2, mlist2 = fft(data2[0:int(len(data1)/2)], freq, fft_w_size)
    # plist3, mlist3 = fft(data3[0:int(len(data1)/2)], freq, fft_w_size)
    # plist4, mlist4 = fft(data4[0:int(len(data1)/2)], freq, fft_w_size)
    #
    # plist1, mlist1 = fft(data1[int(len(data1)/2):len(data1)], freq, fft_w_size)
    # plist2, mlist2 = fft(data2[int(len(data1)/2):len(data1)], freq, fft_w_size)
    # plist3, mlist3 = fft(data3[int(len(data1)/2):len(data1)], freq, fft_w_size)
    # plist4, mlist4 = fft(data4[int(len(data1)/2):len(data1)], freq, fft_w_size)

    # plist1, mlist1 = fft(data1, freq, fft_w_size)
    # plist2, mlist2 = fft(data2, freq, fft_w_size)
    # plist3, mlist3 = fft(data3, freq, fft_w_size)
    # plist4, mlist4 = fft(data4, freq, fft_w_size)
    # print("len", int(len(data1)/2))
    # maxi1 = moving_average_max(mlist1)
    # maxi2= moving_average_max(mlist2)
    # maxi3 = moving_average_max(mlist3)
    # maxi4= moving_average_max(mlist4)
    # # maxi = moving_average_max(np.add(np.add(mlist1, mlist2), np.add(mlist3, mlist4)))
    # mag_start = min([maxi1, maxi2, maxi3, maxi4])
    # mag_end = min([maxi1, maxi2, maxi3, maxi4]) + 3*int(pingc/fft_w_size)
    # mag_end = maxi+int(pingc/fft_w_size)
    # print(int(len(data1)/2), len(data1))
    # print("mag start: ", mag_start, " ", mag_end)
    # plt.figure()
    # plt.plot(mlist1)
    # plt.plot(mlist2)
    # plt.plot(mlist3)
    # plt.plot(mlist4)
    # plt.legend(['1', '2', '3', '4'])
    # plt.title("mag")
    # # plt.figure()
    # # plt.plot(plist1)
    # # plt.plot(plist2)
    # # plt.plot(plist3)
    # # plt.plot(plist4)
    # # plt.legend(['1', '2', '3', '4'])
    # # plt.title("phase")
    # plt.figure()
    # plt.plot(correct_phase(np.subtract(plist2, plist1)))
    # plt.plot(correct_phase(np.subtract(plist3, plist1)))
    # plt.plot(correct_phase(np.subtract(plist4, plist3)))
    # plt.legend(['21', '31', '43'])
    # plt.title("phase diff")
    # plt.show()


    # pdlist12 = np.subtract(plist2, plist1)
    # var12 = variance_list(pdlist12, int(pingc/fft_w_size/2))
    # phase_start12 = moving_average_min(var12[mag_start:mag_end])+mag_start
    # phase_end12 = phase_start12+int(pingc/fft_w_size)
    # pdiff12 = np.mean(correct_phase(pdlist12[phase_start12:phase_end12]))

    # pdiff12 = get_pdiff(plist1, plist2, mag_start, mag_end)
    # pdiff13 = get_pdiff(plist1, plist3, mag_start, mag_end)
    # pdiff34 = get_pdiff(plist3, plist4, mag_start, mag_end)

    # pdiff12, pdiff13, pdiff34 = data_to_pdiff(data1, data2, data3, data4)

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
    # print("\npdiff_12", pdiff12, "pdiff_13", pdiff13, "pdiff_34", pdiff34, "\n")
    # diff12 = pdiff12/2/np.pi*vsound/freq
    # diff13 = pdiff13/2/np.pi*vsound/freq
    # diff34 = pdiff34/2/np.pi*vsound/freq

    # plt.figure()
    # plt.plot(data1_intpw)
    # plt.plot(data2_intpw)
    # plt.plot(data3_intpw)
    # plt.plot(data4_intpw)
    # plt.legend(['1', '2', '3', '4'])
    # plt.show()

    # hp1 = np.array([0, 0, 0])
    # hp2 = np.array([0, -spac, 0])
    # hp3 = np.array([-spac, 0, 0])
    # hp4 = np.array([-spac, -spac, 0])
    # target = np.array([18.59, 27.54, -3.14])
    # target = np.array([2.03, 0.76, -4.72])
    # diff_12 = np.linalg.norm(target-hp1) - np.linalg.norm(target-hp2)
    # diff_13 = np.linalg.norm(target-hp1) - np.linalg.norm(target-hp3)
    # diff_34 = np.linalg.norm(target-hp3) - np.linalg.norm(target-hp4)
    # print(diff_12/vsound*freq*2*np.pi, diff_13/vsound*freq*2*np.pi, diff_34/vsound*freq*2*np.pi)


    # data_val = (hp1, hp2, hp3, hp4, diff12, diff13, diff34)
    # guess = (18, 30, -3)
    # # solution = fsolve(system, guess, args=data_val) #18, 30.5, -3.05
    # solution = solver(guess, diff12, diff13, diff34)
    # x, y, z = solver(guess, diff12, diff13, diff34)
    # ccwh = np.arctan2(y, x)
    # downv = np.arctan2(-z, np.sqrt(x^2 + y^2))
    # print("initial guess", guess)
    # print("x, y, z", x, y, z)
    # print("horizontal angle", ccwh)
    # print("vertical downward angle", downv)
