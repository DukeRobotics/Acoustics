fs=625000;
pingerFrequncy=25000;
clf;

t=linspace(0,2.1,125000*2.1);
wave0=cos(t*2*pi*25000);
wave1=cos(t*2*pi*25000-pi/4);
wave2=cos(t*2*pi*25000-pi/4);
wave0(1:200010)=.001*ones(1, 200010);
wave1(1:200000)=-.001*ones(1, 200000);
wave2(1:200008)=.001*ones(1, 200008);
t2=t(202500:end);
wave0(202500:end)=.1*cos(t2*2*pi*25000+pi/6);
wave1(202500:end)=.1*cos(t2*2*pi*25000+pi/6);
wave2(202500:end)=.1*cos(t2*2*pi*25000+pi/2);
%wave=analog_channel_0;%length must be multiple of 625
[phase0, Fv, time, lengthwave]=phaseForChannel(pingerFrequncy,fs, Channel0);%(1:259375));
[phase1, Fv, time, lengthwave]=phaseForChannel(pingerFrequncy,fs, Channel1);%(1:259375));
[phase2, Fv, time, lengthwave]=phaseForChannel(pingerFrequncy,fs, Channel2);%(1:259375));
% [phase0, Fv, time]=phaseForChannel(pingerFrequncy,fs, wave0(1:259375));
% [phase1, Fv, time]=phaseForChannel(pingerFrequncy,fs, wave1(1:259375));
% [phase2, Fv, time]=phaseForChannel(pingerFrequncy,fs, wave2(1:259375));
arr=zeros(1000);
% for i=321:322
%        [phase0, Fv, time, lengthwave]=phaseForChannel(pingerFrequncy,fs, wave0((625*(i-1)+200):(625*i-1+200)));
%        [phase1, Fv, time, lengthwave]=phaseForChannel(pingerFrequncy,fs, wave1((625*(i-1)+200):(625*i-1+200)));
%        [phase2, Fv, time, lengthwave]=phaseForChannel(pingerFrequncy,fs, wave2((625*(i-1)+200):(625*i-1+200)));
%        bearing=GiveBearing(phase0,phase1,phase2);
%        %arr(i-1)=bearing;
% end

bearing=GiveBearing(phase0,phase1,phase2);
%figure(4)
timeEveryOther=(0:length(wave0)-1)/fs;
%t=linspace(0,2.1,125000*2.1);
timeEveryOther=linspace(0, length(bearing)/fs*2, length(bearing));
%plot( timeEveryOther, bearing)

figure(3)
plot(Channel0)
hold on
plot(Channel1)
hold on
plot(Channel2)