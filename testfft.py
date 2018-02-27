import numpy as np
import math

freq = 40000
freqs = 130000
times = 2
count = times*freqs
signal = []
result = 0
resultf = 0


for i in range(count):
    signal.append(math.cos(2*3.1415926*i*freq/freqs)) #2pi*freq*time, time = i/freqs, 2pi*freq/freqs*i

fft = np.fft.fft(signal)
ffta = np.absolute(fft)

for i in range (35000*times, 45000*times):
    f = i/times;
    if ffta[i] > result:
        resultf = f
        result = ffta[i]
    print fft[i], ffta[i], f

print result, resultf
