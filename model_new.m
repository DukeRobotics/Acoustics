
% p12a = zeros(12, 1);
% p13a = zeros(12, 1);
% p34a = zeros(12, 1);
% for i = 1:100
%     pd = get_pd(i-50);
%     p12a(i) = pd(1);
%     p13a(i) = pd(2);
%     p34a(i) = pd(3);
% end

% plot(p12a)
% hold on
% plot(p13a)
% hold on
% plot(p34a)

%syms x_w y_w z_w
%wall = x_w^2/150^2 + y^2/100^2 + z^2/38^2 == 1;
k = 0;
%function pd = get_pd(k)
    t_len = 2.048*2;
    fs = 625000;
    sample = fs*t_len;
    ft = 30000;
    nl = -76;
    h_dis = 11.5/304.8; % mm to ft
    v_sound = 4858; %ft/s

    % pinger location, first quadrant
    px = 150 - sqrt(17^2-14^2)+k;
    py = 14;
    pz = -sqrt(38^2*(1 - px^2/150^2 - py^2/100^2));

    % hydrophone location, estimation, pointing pos_x direction
    % bot in the top right quadrant in transdec, array mounted on the bot
    % with pinger at the far end, bot just started off the starting deck
    % if not sure about the array location simulated here, ask old members
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
%     hx2 = 150/3 - sqrt(3)*h_dis/2;
%     hy2 = 100/4*3 + h_dis/2;
%     hz2 = -2;
%     hx3 = 150/3;
%     hy3 = 100/4*3 + h_dis;
%     hz3 = -2;
%     hx4 = 150/3 - sqrt(3)*h_dis/4;
%     hy4 = 100/4*3 + h_dis/2;
%     hz4 = -2 - h_dis/4*3;



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

    p1s = ceil(dis1/v_sound*fs)+13000*3;
    p2s = ceil(dis2/v_sound*fs)+13000*3;
    p3s = ceil(dis3/v_sound*fs)+13000*3;
    p4s = ceil(dis4/v_sound*fs)+13000*3;
    p12 = p1s - p2s;
    p13 = p1s - p3s;
    p34 = p3s - p4s;
    pd = [p12, p13, p34];

    off1 = p1s - dis1/v_sound*fs+13000*3;
    off2 = p2s - dis2/v_sound*fs+13000*3;
    off3 = p3s - dis3/v_sound*fs+13000*3;
    off4 = p4s - dis4/v_sound*fs+13000*3;

    s = 1:0.004*fs;
    pingoff1 = (0.1*sin((s+off1)/fs*ft*2*pi))';
    pingoff2 = (0.1*sin((s+off2)/fs*ft*2*pi))';
    pingoff3 = (0.1*sin((s+off3)/fs*ft*2*pi))';
    pingoff4 = (0.1*sin((s+off4)/fs*ft*2*pi))';

    ping1(ceil(dis1/v_sound*fs+13000*3)+1:ceil((dis1/v_sound+0.004)*fs+13000*3)) = pingoff1;
    ping2(ceil(dis2/v_sound*fs+13000*3)+1:ceil((dis2/v_sound+0.004)*fs+13000*3)) = pingoff2;
    ping3(ceil(dis3/v_sound*fs+13000*3)+1:ceil((dis3/v_sound+0.004)*fs+13000*3)) = pingoff3;
    ping4(ceil(dis4/v_sound*fs+13000*3)+1:ceil((dis4/v_sound+0.004)*fs+13000*3)) = pingoff4;

    ping1(ceil(dis1/v_sound*fs+13000*3+2.048*fs)+1:ceil((dis1/v_sound+0.004)*fs+13000*3+2.048*fs)) = 1.5*pingoff1;
    ping2(ceil(dis2/v_sound*fs+13000*3+2.048*fs)+1:ceil((dis2/v_sound+0.004)*fs+13000*3+2.048*fs)) = 1.5*pingoff2;
    ping3(ceil(dis3/v_sound*fs+13000*3+2.048*fs)+1:ceil((dis3/v_sound+0.004)*fs+13000*3+2.048*fs)) = 1.5*pingoff3;
    ping4(ceil(dis4/v_sound*fs+13000*3+2.048*fs)+1:ceil((dis4/v_sound+0.004)*fs+13000*3+2.048*fs)) = 1.5*pingoff4;

    h1 = ping1+wgn(sample, 1, nl);
    h2 = ping2+wgn(sample, 1, nl);
    h3 = ping3+wgn(sample, 1, nl);
    h4 = ping4+wgn(sample, 1, nl);

%end
% h1 = ping1;
% h2 = ping2;
% h3 = ping3;
% h4 = ping4;

% % unit is feet
% function result = wall(x, y, z)
%     result = x^2/150^2 + y^2/100^2 + z^2/38^2;
% end
%
t = table(h1, h2, h3, h4);
t.Properties.VariableNames = {'Channel0' 'Channel1' 'Channel2' 'Channel3'};
writetable(t, '/Users/estellehe/Documents/senior/IndepStudy/matlab_76_4.csv');

function result = dis(x, y, z, x1, y1, z1)
    result = sqrt((x-x1)^2 + (y-y1)^2 + (z-z1)^2);
end
