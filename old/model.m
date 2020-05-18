%syms x_w y_w z_w
%wall = x_w^2/150^2 + y^2/100^2 + z^2/38^2 == 1;

t_len = 4.5;
fs = 625000;
sample = fs*t_len;
ft = 30000;
h_dis = 11.5/304.8; % mm to ft
v_sound = 4858; %ft/s

% pinger location, first quadrant
px = 150 - sqrt(17^2-14^2);
py = 14;
pz = -sqrt(38^2*(1 - px^2/150^2 - py^2/100^2));

% hydrophone location, estimation, pointing pos_x direction
hx1 = 150/3;
hy1 = 100/4*3;
hz1 = -2;
hx2 = 150/3 - h_dis;
hy2 = 100/4*3;
hz2 = -2;
hx3 = 150/3;
hy3 = 100/4*3 + h_dis;
hz3 = -2;
hx4 = 150/3 - h_dis;
hy4 = 100/4*3 + h_dis;
hz4 = -2;


% noise, simulate from data, power shifts from -60db to -73db
noise = wgn(sample, 1, -73);


dis1 = dis(px, py, pz, hx1, hy1, hz1);
dis2 = dis(px, py, pz, hx2, hy2, hz2);
dis3 = dis(px, py, pz, hx3, hy3, hz3);
dis4 = dis(px, py, pz, hx4, hy4, hz4);

% assume pinger ping at t=0 for 4ms, sound arrives at hydrophone with
% amplitude v=5V
ping1 = zeros(sample, 1);
ping2 = zeros(sample, 1);
ping3 = zeros(sample, 1);
ping4 = zeros(sample, 1);
s = 1:0.004*fs;
ping = (0.1*sin(s/fs*ft*2*pi))';

p1s = ceil(dis1/v_sound*fs)+13000*3;
p2s = ceil(dis2/v_sound*fs)+13000*3;
p3s = ceil(dis3/v_sound*fs)+13000*3;
p4s = ceil(dis4/v_sound*fs)+13000*3;

ping1(ceil(dis1/v_sound*fs+13000*3)+1:ceil((dis1/v_sound+0.004)*fs)+13000*3) = ping;
ping2(ceil(dis2/v_sound*fs+13000*3)+1:ceil((dis2/v_sound+0.004)*fs)+13000*3) = ping;
ping3(ceil(dis3/v_sound*fs+13000*3)+1:ceil((dis3/v_sound+0.004)*fs)+13000*3) = ping;
ping4(ceil(dis4/v_sound*fs+13000*3)+1:ceil((dis4/v_sound+0.004)*fs)+13000*3) = ping;

h1 = ping1+wgn(sample, 1, -73);
h2 = ping2+wgn(sample, 1, -73);
h3 = ping3+wgn(sample, 1, -73);
h4 = ping4+wgn(sample, 1, -73);

% % unit is feet
% function result = wall(x, y, z)
%     result = x^2/150^2 + y^2/100^2 + z^2/38^2;
% end

function result = dis(x, y, z, x1, y1, z1)
    result = sqrt((x-x1)^2 + (y-y1)^2 + (z-z1)^2);
end
