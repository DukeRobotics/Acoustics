from matplotlib import pyplot as plt
import numpy as np
import csv
from scipy import signal
import saleae
import time
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
	Matrix=read_file_mat('0-calibrated-625.csv')
	start = 29000
	ra = 2556
	Channel0=np.array(np.asarray(Matrix[0]))[start-1:start+ra-1] #241925:242005  113920:114000
	Channel1=np.array(np.asarray(Matrix[1]))[start-1:start+ra-1]
	Channel2=np.array(np.asarray(Matrix[2]))[start-1:start+ra-1]
	#NEED TO SHORTEN WINDOW--DEPENDS ON USAGE
	b, a=signal.cheby2(3, 3, [((pingerFrequency-8)/fs*2),((pingerFrequency+8)/fs*2)], btype='bandpass')
	FiltChannel0=signal.lfilter(b,a,Channel0)
	FiltChannel1=signal.lfilter(b,a,Channel1)
	FiltChannel2=signal.lfilter(b,a,Channel2)


	diff11=corr(FiltChannel1, FiltChannel0, fs, pingerFrequency) ##if all bearings are opposite of expected, switch order of 2 and 1
	diff12=corr(FiltChannel2, FiltChannel0, fs, pingerFrequency) ##if all bearings are opposite of expected, switch order of 3 and 1


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
		corrArray.append(np.corrcoef(waveA[:n,0], waveB[i:n+i,0])[0][1])
	maxCorr=corrArray.index(max(corrArray)) ##MAY NEED TO SUBTRACT ONE??
	print(corrArray)

	if maxCorr>7.5: ##This should theoretically be 7.09 to 7.1, but experimentally (Due to speed of sound in different water ) is changed to 7.5. 
		maxCorr=maxCorr-maxLag
	if maxCorr<-7.5: ##this should also be -7.09
		#raise("error: not possible") may not need it
	print (maxCorr)
	return maxCorr

def recordFirstListen(host='localhost', port=10429):

	#folder = firstListen
	#os.mkdir(folder)

	s=saleae.Saleae()
	#s=Saleae(host=host, port=port)

	#s.set_num_samples(1e6)

	# for i in range(5):
	#	path = os.path.abspath(os.path.join(folder, str(i)))
	#	s.capture_to_file(path)

	digital = []
	#analog = [0, 1, 2]
	analog = [1]
	s.set_active_channels(digital, analog) #no digital?

	#print(s.get_all_sample_rates())
	s.set_sample_rate((0, 625000))
	s.set_capture_seconds(2.05)

	StartTime=time.time()
	print(StartTime)
	s.capture_start()

	file_path_on_target_machine="/Users/Kelsey/Desktop/Hydrophone/firstListen.csv"


	s.export_data2(file_path_on_target_machine, 
		#digital_channels=None, 
		#analog_channels=[0, 1, 2], 
		format="csv",
		analog_channels=[1],
		analog_format="voltage"
		)

	print(time.time())

	##filter it using normal cheby filter
	##find start of ping
	#find where time of start of ping
	#return this time
	return 0


def getPingerData(host='localhost', port=10429, TimerClock=time.time()):
	currentTime=time.time()
	while currentTime > TimerClock:
		TimerClock+=2 #this should probably 1.9 depending on pinger properties
	time.sleep(TimerClock-currentTime) #may have to add wake up time

	s=saleae.Saleae()
	digital = []
	analog=[0, 1, 2]
	s.set_active_channels(digital, analog)
	s.set_sample_rate((0, 625000))
	s.set_capture_seconds(0.2)
	s.capture_start()
	file_path_on_target_machine="/Users/Kelsey/Desktop/Hydrophone/normalListen.csv"
	s.export_data2(file_path_on_target_machine, 
		digital_channels=None, 
		analog_channels=[0, 1, 2],
		format="csv",
		analog_format="voltage"
		)
	return TimerClock

def mainFunction(pingerFrequency):
	TimeOfPingOffset=recordFirstListen()
	getPingerData(TimerClock=TimeOfPingOffset)
	findPhaseDifference(pingerFrequency, 625000)
	##loop indefinitely getPingerData and findPhase
	## make cases for when you find bad data
	## make cases when you want to restart and find ping offset


