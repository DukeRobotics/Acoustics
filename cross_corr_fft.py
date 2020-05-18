##### to adjust this script to different noise environment,
##### change fft_w_size, large_window_portion, and invalidation requirement in get_pdiff



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

check_len = 20

large_window_portion = 5

gx = 0
gy = 0
gz = -10

hp1 = np.array([0, 0, 0])
hp2 = np.array([0, -spac, 0])
hp3 = np.array([-spac, 0, 0])
hp4 = np.array([-spac, -spac, 0])
# potential future configuration
# hp2 = np.array([-spac/2, -np.sqrt(3)*spac/2, 0])
# hp3 = np.array([-spac, 0, 0])
# hp4 = np.array([-spac/2, -np.sqrt(3)*spac/4, -3*spac/4])


def get_pdiff(parr1, parr2, start, end):
    pdllist = np.subtract(parr2, parr1)
    pdlist = correct_phase(pdllist[start:end])
    # pdlist = large_window(pdslist)
    var = variance_list(pdlist, int(len(pdlist)/large_window_portion))
    # var = variance_list(pdlist, int(len(pdlist)/3))
    phase_start = np.argmin(var)
    # check if lowest variance align with max mag interval, if not then bad data
    # print("phase start", phase_start+start)
    # print("start", start)
    # plt.figure()
    # plt.plot(correct_phase(pdllist))
    # plt.title("phase diff")
    # plt.show()
    ### THIS REQUIREMENT CAN BE LOSEN IF TOO MANY PINGS ARE INVALID, YET MIGHT LEAD TO INACCURATE RESULT
    if phase_start > len(pdlist)/2:
    # if phase_start > len(pdlist)/3*2:
        return None
    phase_end = phase_start+int(len(pdlist)/large_window_portion)
    # phase_end = phase_start+int(len(pdlist)/3)
    pdiff = np.mean(pdlist[phase_start:phase_end])
    return pdiff

def large_window(pdlist):
    result = []
    r = math.ceil(len(pdlist)/large_window_portion)
    for i in range(len(pdlist) - r+1):
        result.append(np.mean(pdlist[i:i+r]))
    return result

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

def moving_average_min(a, n = int(pingc/fft_w_size/10)) :
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
    # plt.title("magnitude")
    # plt.show()

    mlist = np.add(np.add(mlist1, mlist2), np.add(mlist3, mlist4))
    maxi = moving_average_max(mlist)
    mag_start = maxi
    mag_end = maxi + int(pingc/fft_w_size)

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

def check_angle(hz, vt):
    pinger = np.array([check_len*np.cos(vt)*np.cos(hz), check_len*np.cos(vt)*np.sin(hz), -check_len*np.sin(vt)])
    dis1 = np.linalg.norm(pinger-hp1)
    dis2 = np.linalg.norm(pinger-hp2)
    dis3 = np.linalg.norm(pinger-hp3)
    dis4 = np.linalg.norm(pinger-hp4)
    pd12 = (dis1 - dis2)/vsound*freq*2*np.pi
    pd13 = (dis1 - dis3)/vsound*freq*2*np.pi
    pd34 = (dis3 - dis4)/vsound*freq*2*np.pi

    return pd12, pd13, pd34

def plot_3d(ccwh, downv):
    if abs(ccwh) < np.pi/2:
        xline = np.linspace(0, 10, 100)
    else :
        xline = np.linspace(-10, 0, 100)
    yline = xline*np.tan(ccwh)
    zline = -xline/np.cos(ccwh)*np.tan(downv)
    ax.plot3D(xline, yline, zline, 'gray')

def final_hz_angle(hz_arr):
    quadrant = [[], [], [], []]
    for hz in hz_arr:
        if hz >= 0:
            if hz <= np.pi/2:
                quadrant[0].append(hz)
            else:
                quadrant[1].append(hz)
        else:
            if hz <= -np.pi/2:
                quadrant[2].append(hz)
            else:
                quadrant[3].append(hz)
    ave = []
    max_len = np.max([len(each) for each in quadrant])
    for i in range(4):
        if len(quadrant[i]) == max_len:
            ave.extend(quadrant[i])
    return np.mean(ave)


def cross_corr_func(filename, if_double, version, if_plot, samp_f=fs, tar_f=freq, guess_x=gx, guess_y=gy, guess_z=gz):
    global fs, freq, gx, gy, gz, ax
    fs = samp_f
    freq = tar_f
    filepath = filename
    gx = guess_x
    gy = guess_y
    gz = guess_z

    ccwha = []
    downva = []
    xa = []
    ya = []
    za = []
    count = 0

    if version > 0:
        for i in range(version):
            data1, data2, data3, data4 = read_data(filepath.replace(".csv", "("+str(i+1)+").csv"))

            if if_double:
                data1a = [data1[0:int(len(data1)/2)], data1[int(len(data1)/2):len(data1)]]
                data2a = [data2[0:int(len(data1)/2)], data2[int(len(data1)/2):len(data1)]]
                data3a = [data3[0:int(len(data1)/2)], data3[int(len(data1)/2):len(data1)]]
                data4a = [data4[0:int(len(data1)/2)], data4[int(len(data1)/2):len(data1)]]
                double = 2
            else:
                data1a = [data1]
                data2a = [data2]
                data3a = [data3]
                data4a = [data4]
                double = 1

            for j in range(double):
                data1 = data1a[j]
                data2 = data2a[j]
                data3 = data3a[j]
                data4 = data4a[j]
                pdiff12, pdiff13, pdiff34 = data_to_pdiff(data1, data2, data3, data4)

                if (pdiff12 is not None) and (pdiff13 is not None) and (pdiff34 is not None):
                    count = count + 1
                    diff12 = pdiff12/2/np.pi*vsound/freq
                    diff13 = pdiff13/2/np.pi*vsound/freq
                    diff34 = pdiff34/2/np.pi*vsound/freq

                    guess = (gx, gy, gz)
                    x, y, z = solver(guess, diff12, diff13, diff34)
                    xa.append(x)
                    ya.append(y)
                    za.append(z)
                    print("initial guess", guess)
                    print("x, y, z", x, y, z)

                    # calculate angle
                    ccwh = np.arctan2(y, x)
                    ccwha.append(ccwh)
                    print("horizontal angle", ccwh/np.pi*180)
                    downv = np.arctan2(-z, np.sqrt(x**2 + y**2))
                    downva.append(downv)
                    print("vertical downward angle", downv/np.pi*180, "\n")

                    # compare solver result with pdiff from data
                    pd12, pd13, pd34 = check_angle(ccwh, downv)
                    # print("checked pd", pd12, pd13, pd34, "\n")
                    # print("pdiff_12", pdiff12, "pdiff_13", pdiff13, "pdiff_34", pdiff34, "\n")

    final_ccwh = final_hz_angle(ccwha)
    print("\n\nfinal horizontal angle", final_ccwh/np.pi*180)

    if if_plot:
        print("\nsuccess count: ", count, "\n")
        plt.figure()
        ax = plt.axes(projection='3d')
        # ax.scatter3D([-1], [1], [-3]);
        ax.scatter3D([gx], [gy], [gz], color="b");
        #ax.scatter3D([xa], [ya], [za], color="r");
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        for i in range(len(ccwha)):
            plot_3d(ccwha[i], downva[i])
        plt.show()





if __name__ == "__main__":

    cross_corr_func(sys.argv[1], True, 4, True, 625000, 40000, 0, 0, -10)
