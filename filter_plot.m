pingerFreq = 35000;
fs = 625000;
bw = 8;

[b,a]=cheby2(3,2,[(pingerFreq-bw)/fs*2 (pingerFreq+bw)/fs*2], 'bandpass');
filteredWave0 = filter(b, a, Channel0);
filteredWave1 = filter(b, a, Channel1);
filteredWave2 = filter(b, a, Channel2);
filteredWave3 = filter(b, a, Channel3);


figure(1)
plot(filteredWave0)
hold on
plot(filteredWave1)
hold on
plot(filteredWave2)
hold on
plot(filteredWave3)
legend('1', '2', '3', '4');
title('1s');
