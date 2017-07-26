from matplotlib import pyplot as plt
import numpy as np
import csv
from scipy import signal
import saleae


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

def findPhaseDifference(pingerFrequency=25000, fs=625000, ml=16):
	Matrix=read_file_mat('0-calibrated-625.csv')
	Channel1=np.array(np.asarray(Matrix[0]))[113920:114010] #241925:242005  113920:114000
	Channel2=np.array(np.asarray(Matrix[1]))[113920:114010]
	Channel3=np.array(np.asarray(Matrix[2]))[113920:114010]
	b, a=signal.cheby1(2, 3, [((pingerFrequency-20)/fs*2),((pingerFrequency+20)/fs*2)], btype='bandpass')
	FiltChannel1=signal.lfilter(b,a,Channel1)
	FiltChannel2=signal.lfilter(b,a,Channel2)
	FiltChannel3=signal.lfilter(b,a,Channel3)

	diff12=corr(ml, FiltChannel1, FiltChannel2, fs, pingerFrequency)
	diff13=corr(ml, FiltChannel1, FiltChannel3, fs, pingerFrequency)


	bearing=np.arctan2(diff13, diff12)*180/np.pi
	print ("bearing is: %.5f" % bearing)
	return bearing

def corr(ml, waveA, waveB, fs, pingerFrequency):
	n=len(waveA)-ml
	corrArray=[]
	#print(waveA)
	#print(waveB)
	for i in range(ml):
		#print(np.corrcoef(waveA[:n,0], waveB[i:n+i,0]))
		corrArray.append(np.corrcoef(waveA[:n,0], waveB[i:n+i,0])[0][1])
	maxCorr=corrArray.index(max(corrArray))
	print(corrArray)

	if maxCorr>7.062:
		maxCorr=maxCorr-(fs/pingerFrequency)
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


def getPingerData(host='localhost', port=10429, TimerClock=time.time()):
	currentTime=time.time()
	while currentTime > TimerClock:
		TimerClock+=2
	time.sleep(TimerClock-currentTime) #may have to add wake up time

	s=saleae.Saleae()
	digital = []
	analog=[1, 2, 3]
	s.set_active_channels(digital, analog)
	s.set_sample_rate((0, 625000))
	s.set_capture_seconds(0.2)
	s.capture_start()
	file_path_on_target_machine="/Users/Kelsey/Desktop/Hydrophone/normalListen.csv"
	s.export_data2(file_path_on_target_machine, 
		digital_channels=None, 
		analog_channels=[1, 2, 3], 
		format="csv",
		analog_format="voltage"
		)
	return TimerClock

def mainFunction(pingerFrequency):
	TimeOfPingOffset=recordFirstListen()
	getPingerData(TimerClock=TimeOfPingOffset)
	findPhaseDifference(pingerFrequency, 625000)


