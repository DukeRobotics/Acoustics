figure(1)
plot(out1);
hold on
plot(out2);
hold on
plot(out3);
hold on
plot(out4);
legend('1', '2', '3', '4');
title('1s');

[max1, i1] = max(out1);
[max2, i2] = max(out2);
[max3, i3] = max(out3);
[max4, i4] = max(out4);
out = out1+out2+out3+out4;
[max, i] = max(out);
% 
% figure(2)
% plot(out5);
% hold on
% plot(out6);
% hold on
% plot(out7);
% hold on
% plot(out8);
% legend('1', '2', '3', '4');
% title('3s');


% pingerFreq = 40000;
% fs = 625000;
% bw = 8;
% 
% [b,a]=cheby2(3,2,[(pingerFreq-bw)/fs*2 (pingerFreq+bw)/fs*2], 'bandpass');
% filteredWave0 = filter(b, a, Channel0);
% filteredWave1 = filter(b, a, Channel1);
% filteredWave2 = filter(b, a, Channel2);
% filteredWave3 = filter(b, a, Channel3);
% filteredWave4 = filter(b, a, Channel4);
% filteredWave5 = filter(b, a, Channel5);
% filteredWave6 = filter(b, a, Channel6);
% filteredWave7 = filter(b, a, Channel7);
% filteredWave8 = filter(b, a, Channel8);
% filteredWave9 = filter(b, a, Channel9);
% filteredWave10 = filter(b, a, Channel10);
% filteredWave11 = filter(b, a, Channel11);
% filteredWave12 = filter(b, a, Channel12);
% filteredWave13 = filter(b, a, Channel13);
% filteredWave14 = filter(b, a, Channel14);
% filteredWave15 = filter(b, a, Channel15);
% filteredWave16 = filter(b, a, Channel16);
% filteredWave17 = filter(b, a, Channel17);
% filteredWave18 = filter(b, a, Channel18);
% filteredWave19 = filter(b, a, Channel19);
% filteredWave20 = filter(b, a, Channel20);
% filteredWave21 = filter(b, a, Channel21);
% filteredWave22 = filter(b, a, Channel22);
% filteredWave23 = filter(b, a, Channel23);



% figure(1)
% plot(filteredWave0)
% hold on
% plot(filteredWave1)
% hold on
% plot(filteredWave2)
% hold on
% plot(filteredWave3)
% legend('1', '2', '3', '4');
% title('1s');
% 
% figure(2)
% plot(filteredWave4)
% hold on
% plot(filteredWave5)
% hold on
% plot(filteredWave6)
% hold on
% plot(filteredWave7)
% legend('1', '2', '3', '4');
% title('3s');

% figure(3)
% plot(filteredWave8)
% hold on
% plot(filteredWave9)
% hold on
% plot(filteredWave10)
% hold on
% plot(filteredWave11)
% legend('1', '2', '3', '4');
% title('10-2-10');
% 
% figure(4)
% plot(filteredWave12)
% hold on
% plot(filteredWave13)
% hold on
% plot(filteredWave14)
% hold on
% plot(filteredWave15)
% legend('1', '2', '3', '4');
% title('9-1-10');
% 
% figure(5)
% plot(filteredWave16)
% hold on
% plot(filteredWave17)
% hold on
% plot(filteredWave18)
% hold on
% plot(filteredWave19)
% legend('1', '2', '3', '4');
% title('9-1-10');
% 
% figure(6)
% plot(filteredWave20)
% hold on
% plot(filteredWave21)
% hold on
% plot(filteredWave22)
% hold on
% plot(filteredWave23)
% legend('1', '2', '3', '4');
% title('9-2-10');
