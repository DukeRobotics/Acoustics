sf = 625000; % sampling freq
pinger_frequency = 40000;
vs = 1511.5; % velocity of sound
samples = 2.048*2*sf;
noise = -60; % noise level in decibels

%pinger location
px = -2;
py = 7;
pz = 4;

space = 0.0115;  % spacing between hydrophones

%hydrophone location
hp1 = [0, 0, 0];
hp2 = [0, -space, 0];
hp3 = [-space, 0, 0];
hp4 = [-space, -space, 0];

% %cheap hydrophone location
% space = .3;
% hp1 = [0,0,0];
% hp2 = [space, 0,0];
% hp3 = [0, space, 0];
% hp4 = [0, 0, space];

dis1 = dis(px, py, pz, hp1(1), hp1(2), hp1(3));
dis2 = dis(px, py, pz, hp2(1), hp2(2), hp2(3));
dis3 = dis(px, py, pz, hp3(1), hp3(2), hp3(3));
dis4 = dis(px, py, pz, hp4(1), hp4(2), hp4(3));
dis1
dis2
dis3
dis4

% assume pinger ping at t=0 for 4ms, sound arrives at hydrophone with
% amplitude v=5V
ping1 = zeros(samples, 1);
ping2 = zeros(samples, 1);
ping3 = zeros(samples, 1);
ping4 = zeros(samples, 1);

x = 1:0.004*sf;
ping = (0.01*sin(x/sf*pinger_frequency*2*pi))';

buffer = 40000;
ping1(ceil(dis1/vs*sf)+1+buffer:ceil((dis1/vs+0.004)*sf)+buffer) = ping;
ping2(ceil(dis2/vs*sf)+1+buffer:ceil((dis2/vs+0.004)*sf)+buffer) = ping;
ping3(ceil(dis3/vs*sf)+1+buffer:ceil((dis3/vs+0.004)*sf)+buffer) = ping;
ping4(ceil(dis4/vs*sf)+1+buffer:ceil((dis4/vs+0.004)*sf)+buffer) = ping;

ping1(ceil(dis1/vs*sf+2.048*sf)+1+buffer:ceil((dis1/vs+0.004)*sf+2.048*sf)+buffer) = ping;
ping2(ceil(dis2/vs*sf+2.048*sf)+1+buffer:ceil((dis2/vs+0.004)*sf+2.048*sf)+buffer) = ping;
ping3(ceil(dis3/vs*sf+2.048*sf)+1+buffer:ceil((dis3/vs+0.004)*sf+2.048*sf)+buffer) = ping;
ping4(ceil(dis4/vs*sf+2.048*sf)+1+buffer:ceil((dis4/vs+0.004)*sf+2.048*sf)+buffer) = ping;

for i = 1:4

    h1 = ping1+wgn(samples, 1, noise);
    h2 = ping2+wgn(samples, 1, noise);
    h3 = ping3+wgn(samples, 1, noise);
    h4 = ping4+wgn(samples, 1, noise);

    t = table(h1, h2, h3, h4);
    t.Properties.VariableNames = {'Channel 0', 'Channel 1', 'Channel 2', 'Channel 3'};
    
    filepath = sprintf('/Users/reedchen/OneDrive - Duke University/Robotics/Data/matlab_custom_-2_7_4_(%d).csv', i);
    writetable(t, filepath);
    disp(i)
end

plot(t.('Channel 0'));
hold on
plot(t.('Channel 1'));
hold on
plot(t.('Channel 2'));
hold on
plot(t.('Channel 3'));


function result = dis(x, y, z, x1, y1, z1)
    result = sqrt((x-x1)^2 + (y-y1)^2 + (z-z1)^2);
end