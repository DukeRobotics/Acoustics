import argparse
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
from scipy.optimize import fsolve
import cmath

# default command line arguments
verbose = 1
#filepath, files  = "/Users/reedchen/OneDrive - Duke University/Robotics/Data/625k_40k_4_-5_-4.csv", [1,2,3,4]
#filepath, files  = "/Users/reedchen/OneDrive - Duke University/Robotics/Data/matlab_custom.csv", [1,2,3,4]
#filepath, files = "/Users/reedchen/OneDrive - Duke University/Robotics/Data/matlab_custom_001_.csv", [1,2,3,4]
#filepath, files = "/Users/reedchen/OneDrive - Duke University/Robotics/Data/CompetitionData/625k_30k_1_-1_10.csv", [1,2,3]
#filepath, files = "/Users/reedchen/OneDrive - Duke University/Robotics/Data/CompetitionData/625k_30k_1_1_10.csv", [7]

sampling_time = 3  # length of data in seconds
pinger_timing = 2  # seconds between pings
fourier_width = 125  # number of samples to perform discrete fourier transform on
target_frequency = 40000  # frequency of pinger, 30000 for competition, 40000 for pool, 40000 for sink
sampling_frequency = 625000  # samples taken per second
ping_duration = .004  # how long the ping lasts in seconds
vsound = 1511.5  # speed of sound in water

# guessed pinger location
gx = 1
gy = -1
gz = 10

# space = 0.0115  # spacing between hydrophones
# hp0 = [0, 0, 0]
# hp1 = [0, -space, 0]
# hp2 = [-space, 0, 0]
# hp3 = [-space, -space, 0]

hp0 = [0, 0, 0]
hp1 = [0, -18.42878e-3, -11.37982e-3]
hp2 = [12.72792e-3, 12.72792e-3, 0]
hp3 = [-12.72792e-3, 12.72792e-3, 0]

filepath, files = "/Users/reedchen/OneDrive - Duke University/Robotics/Data/SinkData/HPsink2.csv", [1]

gx = -5
gy = -5
gz = 0

def read_data(filepath):
    df = pd.read_csv(filepath, skiprows=[1], skipinitialspace=True)
    c1 = df["Channel 0"].tolist()
    c2 = df["Channel 1"].tolist()
    c3 = df["Channel 2"].tolist()
    c4 = df["Channel 3"].tolist()
    return c1, c2, c3, c4


def split_data(rawData, split):
    chunkLength = len(rawData) // split
    return [rawData[i * chunkLength:(i + 1) * chunkLength] for i in range(split)]


def dft(data):
    phase = []
    mag = []
    for i in range(len(data) // fourier_width):
        batch = data[i * fourier_width:(i + 1) * fourier_width]
        n = np.array(range(fourier_width))
        exp = np.exp(-1j * 2 * np.pi * n * target_frequency / sampling_frequency)
        transform = np.dot(batch, exp)
        phase.append(cmath.phase(transform))
        mag.append(np.abs(transform))
    return phase, mag


def max_moving_avg(data, window):
    max = 0
    maxIndex = 0
    kernel = [1 / window] * window
    alist = np.convolve(data, kernel, mode="valid")
    for i in range(len(alist)):
        if alist[i] > max:
            max = alist[i]
            maxIndex = i
    return maxIndex


def correct_phase(phaseDiff):
    correctedPhase = []
    for phase in phaseDiff:
        if abs(phase) > np.pi:
            correctedPhase.append(-phase / abs(phase) * (2 * np.pi - abs(phase)))
        else:
            correctedPhase.append(phase)
    return correctedPhase


def var_list(data, window):
    varList = []
    for i in range(len(data) - window):
        varList.append(np.var(data[i:i + window]))
    return varList


def find_phase_diff(phase1, phase2, start, end):
    phaseDiff = np.subtract(phase2, phase1)
    correctedPhase = correct_phase(phaseDiff)

    if verbose >= 2:
        plt.plot(correctedPhase, label="Corrected")
        plt.plot(phaseDiff, label="Uncorrected", linestyle="--")
        plt.axvline(x=start, color="g", linestyle="--", linewidth=.5, label="min variance start")
        plt.axvline(x=end - 1, color="r", linestyle="--", linewidth=.5, label="min variance end")
        plt.title("Phase Difference")
        plt.legend()
        plt.show()

    return np.mean(correctedPhase[start:end])

def system(guess, *variables):
    hp0, hp1, hp2, hp3, timeDiff01, timeDiff02, timeDiff03 = variables
    return (np.linalg.norm(guess - hp0) - np.linalg.norm(guess - hp1) - timeDiff01,
            np.linalg.norm(guess - hp0) - np.linalg.norm(guess - hp2) - timeDiff02,
            np.linalg.norm(guess - hp0) - np.linalg.norm(guess - hp3) - timeDiff03)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", "-f", help="filepath to data")
    parser.add_argument("--verbose", "-v", help="display graphs, 0 (no graphs) to 3 (all graphs)")
    args = parser.parse_args()
    if args.filepath is not None:
        filepath = args.filepath
    if args.verbose is not None:
        verbose = args.verbose

    xLocations = []
    yLocations = []
    zLocations = []

    for i in files:
        rawData0, rawData1, rawData2, rawData3 = read_data(filepath.replace(".csv", "(" + str(i) + ").csv"))

        if verbose >= 3:
            plt.plot(rawData0)
            plt.plot(rawData1)
            plt.plot(rawData2)
            plt.plot(rawData3)
            plt.title(f"Raw data from file {i}")
            plt.xlabel("Samples")
            plt.ylabel("Voltage")
            plt.legend(["Channel 0", "Channel 1", "Channel 2", "Channel 3"], loc="upper right")
            plt.show()

        split = math.ceil(sampling_time / pinger_timing)
        splitData0 = split_data(rawData0, split)
        splitData1 = split_data(rawData1, split)
        splitData2 = split_data(rawData2, split)
        splitData3 = split_data(rawData3, split)

        for j in range(split):
            phase0, mag0 = dft(splitData0[j])
            phase1, mag1 = dft(splitData1[j])
            phase2, mag2 = dft(splitData2[j])
            phase3, mag3 = dft(splitData3[j])

            if verbose >= 3:
                fig, axs = plt.subplots(2)
                axs[0].plot(phase0)
                axs[0].plot(phase1)
                axs[0].plot(phase2)
                axs[0].plot(phase3)
                axs[0].set_title(f"Phases from file {i} batch {j}")
                axs[1].plot(mag0)
                axs[1].plot(mag1)
                axs[1].plot(mag2)
                axs[1].plot(mag3)
                axs[1].set_title(f"Magnitudes from file {i} batch {j}")
                for ax in axs.flat:
                    ax.set(xlabel='x', ylabel='y')
                    ax.label_outer()
                    ax.legend(["Channel 0", "Channel 1", "Channel 2", "Channel 3"], loc="upper right")
                plt.show()

            totalMag = np.add(np.add(mag0, mag1), np.add(mag2, mag3))
            samplesPerPing = sampling_frequency * ping_duration
            window = math.floor(samplesPerPing / fourier_width)
            maxIndex = max_moving_avg(totalMag, window)
            maxIndexEnd = maxIndex + window

            if verbose >= 1:
                fig, axs = plt.subplots(5)
                axs[0].plot(totalMag)
                axs[1].plot(phase0)
                axs[2].plot(phase1)
                axs[3].plot(phase2)
                axs[4].plot(phase3)
                for k in range(5):
                    axs[k].axvline(x=maxIndex, color="g", linestyle="--", linewidth=.5)
                    axs[k].axvline(x=maxIndexEnd - 1, color="r", linestyle="--", linewidth=.5)
                    axs[k].set_title(f"Channel {k - 1} phase")
                    axs[k].legend(["phase", "max index start", "max index end"], loc="upper right")
                axs[0].set_title("total mag")
                fig.tight_layout()
                plt.show()

            phase0 = phase0[maxIndex:maxIndexEnd]
            phase1 = phase1[maxIndex:maxIndexEnd]
            phase2 = phase2[maxIndex:maxIndexEnd]
            phase3 = phase3[maxIndex:maxIndexEnd]

            varWindow = window // 5
            varList0 = var_list(phase0, varWindow)
            varList1 = var_list(phase1, varWindow)
            varList2 = var_list(phase2, varWindow)
            varList3 = var_list(phase3, varWindow)
            totalVar = np.add(np.add(varList0, varList1), np.add(varList2, varList3))
            minIndex = np.argmin(totalVar)
            minIndexEnd = minIndex + varWindow

            if verbose >= 1:
                fig, axs = plt.subplots(4)
                axs[0].plot(phase0)
                axs[1].plot(phase1)
                axs[2].plot(phase2)
                axs[3].plot(phase3)
                for k in range(4):
                    axs[k].axvline(x=minIndex, color="g", linestyle="--", linewidth=.5)
                    axs[k].axvline(x=minIndexEnd, color="r", linestyle="--", linewidth=.5)
                    axs[k].set_title(f"Channel {k} phase")
                    axs[k].legend(["phase", "min index start", "min index end"], loc="upper right")
                fig.tight_layout()
                plt.show()

            phaseDiff01 = find_phase_diff(phase0, phase1, minIndex, minIndexEnd)
            phaseDiff02 = find_phase_diff(phase0, phase2, minIndex, minIndexEnd)
            phaseDiff03 = find_phase_diff(phase0, phase3, minIndex, minIndexEnd)

            diff01 = phaseDiff01 / 2 / np.pi * vsound / target_frequency
            #print(7.5422-7.5498)
            print(diff01)
            print()
            diff02 = phaseDiff02 / 2 / np.pi * vsound / target_frequency
            #print(7.5559 - 7.5498)
            print(diff02)
            print()
            diff03 = phaseDiff03 / 2 / np.pi * vsound / target_frequency
            #print(7.5483 - 7.5498)
            print(diff03)
            print()

            guessedLocation = (gx, gy, gz)

            data_val = (hp0, hp1, hp2, hp3, diff01, diff02, diff03)
            x, y, z = fsolve(system, guessedLocation, data_val)

            print(f"x: {x}\ty: {y}\tz: {z}")

            xLocations.append(x)
            yLocations.append(y)
            zLocations.append(z)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(xLocations)):
        norm = math.sqrt(xLocations[i]**2 + yLocations[i]**2 + zLocations[i]**2)
        ax.plot([0, xLocations[i] / norm * 10], [0, yLocations[i] / norm * 10], [0, zLocations[i] / norm * 10])
    ax.scatter([gx], [gy], [gz])
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.axes.set_xlim3d(left=-10, right=10)
    ax.axes.set_ylim3d(bottom=-10, top=10)
    ax.axes.set_zlim3d(bottom=-10, top=10)
    plt.show()
