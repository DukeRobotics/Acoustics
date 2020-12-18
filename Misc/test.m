pingerFreq = 30000;
fs = 625000;
startP = 783600;
range = 600;

[b,a]=cheby2(3,3,[(pingerFreq-8)/fs*2 (pingerFreq+8)/fs*2], 'bandpass');
filteredWave0 = filter(b, a, Channel0);
filteredWave1 = filter(b, a, Channel1);
filteredWave2 = filter(b, a, Channel2);


Channel0Short = filteredWave0(startP:startP+range);%1077520:1078020 with error in 180-ca-2
Channel1Short = filteredWave1(startP:startP+range);  %128950:129030
Channel2Short = filteredWave2(startP:startP+range); %113920:114000   241925:242005

figure(3)
plot(Channel0Short)
hold on
plot(Channel1Short)
hold on
plot(Channel2Short)
maxlag = fs/pingerFreq;
n = length(Channel0Short)-fix(maxlag);
corArr1 = zeros(1,fix(maxlag+1));
for i = 0:(fix(maxlag-1))
    temp=corrcoef(Channel1Short(1:n+1), Channel0Short(i+1:n+i+1));
    corArr1(i+1) =temp(1, 2);
end%
maxCor1 = find(corArr1 == max(corArr1))-1;
if maxCor1>7.095; %should be 7.095 but experimentally changed to 7.5
    maxCor1=maxCor1-maxlag;
end

corArr2 = zeros(1,fix(maxlag+1));
for i = 0:(fix(maxlag-1));
    temp=corrcoef(Channel2Short(1:n+1), Channel0Short(i+1:n+i+1));
    corArr2(i+1) =temp(1, 2);
end
maxCor2 = find(corArr2 == max(corArr2))-1;
if maxCor2>7.095
    maxCor2=maxCor2-maxlag;
end
if maxCor2<-7.7 ||maxCor1<-7.7;
    %throw('error')
end

atan2d(maxCor1, maxCor2)