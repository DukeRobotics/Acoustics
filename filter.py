from scipy.signal import cheby2, lfilter
import csv
import matplotlib.pyplot as plt
import subprocess

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

if __name__ == "__main__":
    data = []
    with open("data.csv", 'rb') as filec:
        reader = csv.reader(filec)
        for row in reader:
            try:
                p = float(row[0])
                data.append(float(row[0]))
            except:
                continue
    try:
        out = cheby2_bandpass_filter(data, 40000, 60000, 120000)
    except Exception as e:
        print e
    #print out[0]
    with open("out.csv", 'wb') as write:
        writer = csv.writer(write)
        for point in out:
            writer.writerow([round(point, 4)])
    plt.plot(data)
    #plt.plot(out)
    subprocess.call(["rm", "testcsv"])
    subprocess.call(["gcc", "-o", "testcsv", "testcsv.c", "-lfftw3", "-lm"])
    subprocess.call("./testcsv")
    plt.show()
