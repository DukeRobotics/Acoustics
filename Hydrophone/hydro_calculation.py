import numpy as np
import csv
from scipy import signal
#import saleae
#import time
import math


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


def findPhaseDifference(pingerFrequency=25000, fs=625000):
    Matrix=read_file_mat('west-ca-1.csv')
    start = 29000
    ra = 2556
    Channel1=np.array(np.asarray(Matrix[0]))[start-1:start+ra-1] #241925:242005  113920:114000
    Channel2=np.array(np.asarray(Matrix[1]))[start-1:start+ra-1]
    Channel3=np.array(np.asarray(Matrix[2]))[start-1:start+ra-1]
    #NEED TO SHORTEN WINDOW--DEPENDS ON USAGE
    b, a=signal.cheby2(3, 3, [((pingerFrequency-8)/fs*2),((pingerFrequency+8)/fs*2)], btype='bandpass')
    FiltChannel1=signal.lfilter(b,a,Channel1)
    FiltChannel2=signal.lfilter(b,a,Channel2)
    FiltChannel3=signal.lfilter(b,a,Channel3)

    diff12=corr(FiltChannel2, FiltChannel1, fs, pingerFrequency) ##if all bearings are opposite of expected, switch order of 2 and 1
    diff13=corr(FiltChannel3, FiltChannel1, fs, pingerFrequency) ##if all bearings are opposite of expected, switch order of 3 and 1

    bearing=np.arctan2(diff12, diff13)*180/np.pi
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
		corrArray.append(np.corrcoef(waveA[:n,0], waveB[i:n+i,0])[0][1])
	maxCorr=corrArray.index(max(corrArray)) ##MAY NEED TO SUBTRACT ONE??
	print(corrArray)

	if maxCorr>7.09: ##This should theoretically be 7.09 to 7.1, but experimentally (Due to speed of sound in different water ) is changed to 7.5.
		maxCorr=maxCorr-maxLag
	#if maxCorr<-7.09: ##this should also be -7.09
		#raise("error: not possible")   may not need it
	print (maxCorr)
	return maxCorr

 print (findPhaseDifference(40000, 625000))