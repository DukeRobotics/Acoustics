function [ phase, Fv, time, lengthwave] = phaseForChannel( pingerFreq, fs, wave )
%PHASEFORCHANNEL Summary of this function goes here
%   Detailed explanation goes here

[b,a]=cheby2(3,3,[(pingerFreq-8)/fs*2 (pingerFreq+8)/fs*2], 'bandpass');
%filteredWave=filter(b,a,wave);
filteredWave=filter(b,a,wave);
%filteredWave=filteredWave(181750:182374);
%filteredWave=filteredWave(102050:102674);
time=(0:length(filteredWave)-1)/fs;
lengthwave=length(filteredWave);
% length1=length(filteredWave);
% res=fft(filteredWave, length1)/length1;
% res0=res(1:length1/2+1);
%disp(filteredWave(1:10))
%divisor=fs/length(time);
%lengthOfRes0=length(res0);
%phase=180/pi*angle(res0(25000/divisor+1));



Fs=fs;                                                % Sampling Frequency
Fn = Fs/2;                                                  % Nyquist Frequency
s=filteredWave;
L  = length(s);
fts = fft(s)/L;                                            % Normalised Fourier Transform
fts=fts(2:end);
%figure(2);
%plot (fts)
first=angle(fts);
first(1);
fts(1:10);
Fv = linspace(0, 1, fix(L/2)+1)*Fn;                         % Frequency Vector  
Iv = 1:length(Fv);                                          % Index Vector

amp_fts = abs(fts(Iv))*2;                                   % Spectrum Amplitude
phs_fts = angle(fts(Iv));                                   % Spectrum Phase
phase=phs_fts;

% threshold= max(abs(fts))/10000;
% for i=1:length(fts);
%     if fts(i)<threshold;;
%         fts(i)=0;
%     end
% end
% phase=atan2(imag(fts), real(fts));
% plot(phase)
% 
 figure(1); hold on
%res=fft(filteredWave);
%freq=0:fs/length(filteredWave):fs/2;
%mag=20*log10(abs(res));
%ph=(180/pi)*unwrap(angle(res));
%ph=180/pi*(angle(res0));
plot(filteredWave)
% figure(2)
% plot(freq,ph)


end