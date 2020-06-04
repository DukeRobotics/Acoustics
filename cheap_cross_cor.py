import numpy as np
import pandas as pd
from scipy.signal import correlate
import matplotlib.pyplot as plt

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

octant = [0, 0, 0]


def main():
    df = pd.read_csv(filepath)
    c1 = df["Channel 0"].tolist() # 1323120
    c2 = df["Channel 1"].tolist() # 1323060
    c3 = df["Channel 2"].tolist()
    c4 = df["Channel 3"].tolist()

    cross12 = correlate(c1, c2, mode='full') # 2560060 - 2560000 = 60. c1 is 60 samples later than c2
    cross13 = correlate(c1, c3, mode='full')
    cross14 = correlate(c1, c4, mode='full')

    x = np.argmax(np.array(cross12)) - len(c1) # positive
    y = np.argmax(np.array(cross13)) - len(c1) # negative
    z = np.argmax(np.array(cross14)) - len(c1) # negative

    print(x, y, z)

    octant[0] = 1 if x > 0 else -1
    octant[1] = 1 if y > 0 else -1
    octant[2] = 1 if z > 0 else -1


def plot(data):
    fig, axs = plt.subplots(len(data))
    for i, ax in enumerate(axs.flat):
        ax.plot(data[i])
    plt.show()


if __name__ == '__main__':
    main()