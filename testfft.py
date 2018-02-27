import numpy as np
import math

freq = 40000
freqs = 130000
times = 2
count = times*freqs
signal = []
result = 0
resultf = 0
start = 2
end = 130245


for i in range(count):
    signal.append(math.cos(2*3.1415926*i*freq/freqs)) #2pi*freq*time, time = i/freqs, 2pi*freq/freqs*i



dataw = signal[start:end+1]
time = np.linspace(start/freqs, end/freqs, end-start+1)

#fft
fft = np.fft.fft(dataw)
ffta = np.absolute(fft)
result = 0
resultf = 0

timew = (end-start+1)/float(freqs)
print timew
#find max
for i in range(len(ffta)):
    f = i/timew;
    if ffta[i] > result:
        resultf = f
        result = ffta[i]
    print fft[i], ffta[i], f

print result, resultf
