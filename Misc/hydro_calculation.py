import numpy as np
import csv
from scipy import signal
#import saleae
#import time
import math
#import matplotlib.pyplot as plt


def read_file_mat(filename, numHeaders=2):
	reader=csv.reader(open(filename), delimiter=',')
	total = [[],[],[],[]]
	for row in reader:
		try:
			for i in range(len(row)):
				total[i].append([float(row[i])])
		except:
			pass
	channels = [col for col in total[1:len(total)]]
	return channels

def movingAverage(arr, n):
	return np.append(np.array([np.mean(arr[:i + 1]) for i in range(n - 1)]),
                     np.correlate(arr, np.ones(n, dtype=float) / n))

def detectWavefront(signals, pingerFrequency, fs):
    # tune these two constants to make it more robust / reduce lag (increasing sn will
    # reduce false positives, but increase lag)
    signals = filterWave(signals, pingerFrequency, fs)
    #print(signals[0:10])
    n = 10000
    sn = 1000
    thresh = 3.9 # tune this to similarly adjust robustness
    amplitude = np.abs(signals)
    longma = movingAverage(amplitude, n)
    shortma = movingAverage(amplitude, sn)
    return np.argmax(shortma > thresh * longma)

def filterWave(signals, pingerFrequency=40000, fs=625000):
	b, a = signal.cheby2(3, 3, [((pingerFrequency - 8) / fs * 2), ((pingerFrequency + 8) / fs * 2)], btype='bandpass')
	filteredwave = signal.lfilter(b, a, signals, axis=0)
	filteredwave = [x for n in filteredwave for x in n]
	return filteredwave

def findPhaseDifference(filename, add, pingerFrequency=40000, fs=625000):
	Matrix=read_file_mat(filename)
	ra = 200


	Channel0=np.array(np.asarray(Matrix[0]))
	Channel1=np.array(np.asarray(Matrix[1]))
	Channel2=np.array(np.asarray(Matrix[2]))
    #NEED TO SHORTEN WINDOW--DEPENDS ON USAGE
	ping0 = detectWavefront(Channel0, pingerFrequency, fs)     #move 300 sample ahead to get the true wavefront, from experience
	ping1 = detectWavefront(Channel1, pingerFrequency, fs)
	ping2 = detectWavefront(Channel2, pingerFrequency, fs)
	ping = min([ping0, ping1, ping2])
	end = len(Channel0)

	filtChannel0 = filterWave(Channel0, pingerFrequency, fs)
	filtChannel1 = filterWave(Channel1, pingerFrequency, fs)
	filtChannel2 = filterWave(Channel2, pingerFrequency, fs)

	Channel0s = filtChannel0[ping+add-1:ping+add+ra-1]
	Channel1s = filtChannel1[ping + add - 1:ping+add+ra - 1]
	Channel2s = filtChannel2[ping + add - 1:ping+add+ra - 1]
	#b, a=signal.cheby2(3, 3, [((pingerFrequency-8)/fs*2),((pingerFrequency+8)/fs*2)], btype='bandpass')
	#FiltChannel0=signal.lfilter(b,a,Channel0, axis=0)
	#FiltChannel1=signal.lfilter(b,a,Channel1, axis=0)
	#FiltChannel2=signal.lfilter(b,a,Channel2, axis=0)
	#FiltChannel0 = filterWave(Channel0s, pingerFrequency, fs)
	#FiltChannel1 = filterWave(Channel1s, pingerFrequency, fs)
	#FiltChannel2 = filterWave(Channel2s, pingerFrequency, fs)

	print(ping+add)

	#print(fChannel0s[0:10])
	diff11=corr(Channel1s, Channel0s, fs, pingerFrequency) ##if all bearings are opposite of expected, switch order of 2 and 1
	diff12=corr(Channel2s, Channel0s, fs, pingerFrequency) ##if all bearings are opposite of expected, switch order of 3 and 1

	bearing=np.arctan2(diff11, diff12)*180/np.pi
	print ("bearing is: %.5f" % bearing)
	return bearing

def corr(waveA, waveB, fs, pingerFrequency):
	maxLag=fs/pingerFrequency
	n=len(waveA)-math.floor(maxLag)
	corrArray=[]
	#print(waveA)
	#print(waveB)
	for i in range(math.floor(maxLag-1)): #DO YOU NEED TO SUBTRACT ONE?
		#print(np.corrcoef(waveA[:n,0], waveB[i:n+i,0]))
		corrArray.append(np.corrcoef(waveA[:n], waveB[i:n+i])[0][1])
	maxCorr=corrArray.index(max(corrArray)) ##MAY NEED TO SUBTRACT ONE??
	#print(corrArray)

	if maxCorr>7.09: ##This should theoretically be 7.09 to 7.1, but experimentally (Due to speed of sound in different water ) is changed to 7.5.
		maxCorr=maxCorr-maxLag
	#if maxCorr<-7.09: ##this should also be -7.09
		#raise("error: not possible")   may not need it
	print (maxCorr)
	return maxCorr


def findPing(wave, fs = 625000):
	window = fs/625000*2000
	maxPing = max(wave)
	maxID = wave.tolist().index(maxPing)
	result = maxID - window/2
	return result, maxPing


def bearing(filename, pingerFrequency = 25000, fs = 625000):
	bearinglist = []
	for i in range(1):
		i = i*100
		bearinglist.append(findPhaseDifference(filename, i, pingerFrequency, fs))
	return (sum(bearinglist)/len(bearinglist))


print(bearing('180-625-8-25khz.csv', 25000, 625000))