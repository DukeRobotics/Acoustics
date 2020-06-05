import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz, correlate

'''
pinger location
px = 4;
py = -5;
pz = -4;

cheap hydrophone location
space = .3;
hp1 = [0,0,0];
hp2 = [space, 0,0];
hp3 = [0, space, 0];
hp4 = [0, 0, space];
'''

filepath = '../Data/matlab_custom_cheap_hydrophone_(1).csv'

fs = 625000
pinger_frequency = 40000
band_width = 500

octant = [0, 0, 0]


def main():
    df = pd.read_csv(filepath)
    c1 = df["Channel 0"].tolist() # 1323120
    c2 = df["Channel 1"].tolist() # 1323060
    c3 = df["Channel 2"].tolist()
    c4 = df["Channel 3"].tolist()

    lowcut = pinger_frequency-band_width//2
    highcut = pinger_frequency+band_width//2

    #plot_filter(lowcut, highcut, fs, 4)

    f1 = butter_bandpass_filter(c1, lowcut, highcut, fs, order=4)
    f2 = butter_bandpass_filter(c2, lowcut, highcut, fs, order=4)
    f3 = butter_bandpass_filter(c3, lowcut, highcut, fs, order=4)
    f4 = butter_bandpass_filter(c4, lowcut, highcut, fs, order=4)

    print(len(f1))
    plot([c1, f1])

    cross12 = correlate(f1, f2, mode='full') # 2560060 - 2560000 = 60. c1 is about 60 samples later than c2
    cross13 = correlate(f1, f3, mode='full')
    cross14 = correlate(f1, f4, mode='full')

    x = np.argmax(np.array(cross12)) - len(f1) # positive
    y = np.argmax(np.array(cross13)) - len(f1) # negative
    z = np.argmax(np.array(cross14)) - len(f1) # negative

    print(x, y, z)

    octant[0] = 1 if x > 0 else -1
    octant[1] = 1 if y > 0 else -1
    octant[2] = 1 if z > 0 else -1


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def plot_filter(lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    w, h = freqz(b, a, worN=2000)
    plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)
    plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)], '--', label='sqrt(0.5)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.grid(True)
    plt.legend(loc='best')
    plt.show()


def plot(data):
    fig, axs = plt.subplots(len(data))
    for i, ax in enumerate(axs.flat):
        ax.plot(data[i])
    plt.show()


if __name__ == '__main__':
    main()