v0 = interp1(x0, 1:0.1:length(x0), 'pchip');
v1 = interp1(x1, 1:0.1:length(x1), 'pchip');
v2 = interp1(x2, 1:0.1:length(x2), 'pchip');
figure(1)
plot(v0)
hold on
plot(v1)
hold on
plot(v2)

r1 = xcorr(v0(319850:319910), v1(319850:319910));
r2 = xcorr(v0(319850:319910), v2(319850:319910));
r3 = xcorr(v1(319850:319910), v2(319850:319910));
%plot(r)
[ma1, loc1] = max(r1);
[ma2, loc2] = max(r2);
[ma3, loc3] = max(r3);
