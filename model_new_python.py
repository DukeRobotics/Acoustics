import pandas
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math as math
import numpy as np
from astropy.table import QTable, Table, Column
from tabulate import tabulate
import os.path
import sys

def dis(x, y, z, x1, y1, z1):
    return math.sqrt((x - x1) ** 2 + (y - y1) ** 2 + (z - z1) ** 2)


sf = 625000  # sampling freq
pinger_frequency = 40000
vs = 1511.5  # velocity of sound
samples = 2.048 * 2 * sf
noise = -60  # noise level in decibels

# pinger location
px = -2
py = 7
pz = 4

space = 0.0115  # spacing between hydrophones

# hydrophone location
hp1 = [0, 0, 0]
hp2 = [0, -space, 0]
hp3 = [-space, 0, 0]
hp4 = [-space, -space, 0]

# cheap hydrophone location
# space = .3
# hp1 = [0, 0, 0]
# hp2 = [space, 0, 0]
# hp3 = [0, space, 0]
# hp4 = [0, 0, space]

dis1 = dis(px, py, pz, hp1[0], hp1[1], hp1[2])
dis2 = dis(px, py, pz, hp2[0], hp2[1], hp2[2])
dis3 = dis(px, py, pz, hp3[0], hp3[1], hp3[2])
dis4 = dis(px, py, pz, hp4[0], hp4[1], hp4[2])
# dis1
# dis2
# dis3
# dis4

# assume pinger ping at t = 0 for 4ms, sound arrives at hydrophone with amplitude v = 5V
ping1 = np.zeros(math.floor(samples))
ping2 = np.zeros(math.floor(samples))
ping3 = np.zeros(math.floor(samples))
ping4 = np.zeros(math.floor(samples))

s = np.arange(1, math.floor(0.004 * sf), 1/sf)
ping = 0.1 * np.sin((s / sf * pinger_frequency * 2 * math.pi))  # complex conjugate transpose (np.matrix.H)

buffer = 40000

# sorry, i will make this part more efficient
for i in range(math.ceil(dis1 / vs * sf) + 1 + buffer, math.ceil((dis1 / vs + 0.004) * sf) + buffer + 1):
    ping1[i] = ping[i]

for i in range(math.ceil(dis2 / vs * sf) + 1 + buffer, math.ceil((dis2 / vs + 0.004) * sf) + buffer + 1):
    ping2[i] = ping[i]

for i in range(math.ceil(dis3 / vs * sf) + 1 + buffer, math.ceil((dis3 / vs + 0.004) * sf) + buffer + 1):
    ping3[i] = ping[i]

for i in range(math.ceil(dis4 / vs * sf) + 1 + buffer, math.ceil((dis4 / vs + 0.004) * sf) + buffer + 1):
    ping4[i] = ping[i]

for i in range(math.ceil(dis1 / vs * sf + 2.048 * sf) + 1 + buffer,
               math.ceil((dis1 / vs + 0.004) * sf + 2.048 * sf) + buffer + 1):
    ping1[i] = ping[i]

for i in range(math.ceil(dis2 / vs * sf + 2.048 * sf) + 1 + buffer,
               math.ceil((dis2 / vs + 0.004) * sf + 2.048 * sf) + buffer + 1):
    ping2[i] = ping[i]

for i in range(math.ceil(dis3 / vs * sf + 2.048 * sf) + 1 + buffer,
               math.ceil((dis3 / vs + 0.004) * sf + 2.048 * sf) + buffer + 1):
    ping3[i] = ping[i]

for i in range(math.ceil(dis4 / vs * sf + 2.048 * sf) + 1 + buffer,
               math.ceil((dis4 / vs + 0.004) * sf + 2.048 * sf) + buffer + 1):
    ping4[i] = ping[i]

for i in range(1, 5):
    mean = (-76 + noise) / 2
    std = abs((-76 - noise) / 6)

    h1 = ping1 + np.random.normal(mean, std, samples)
    h2 = ping2 + np.random.normal(mean, std, samples)
    h3 = ping3 + np.random.normal(mean, std, samples)
    h4 = ping4 + np.random.normal(mean, std, samples)

    t = Table([h1, h2, h3, h4], names=('Channel 0', 'Channel 1', 'Channel 2', 'Channel 3'))
    with open(('/Users/reedchen/OneDrive - Duke University/Robotics/Data/matlab_custom_-2_7_4_(%d).csv', i),
              'w') as outputfile:
        outputfile.write(tabulate(t))

fig, axs = plt.subplots(4)
fig.subtitle('Sine Waves')
axs[0].plot(h1)
axs[1].plot(h2)
axs[2].plot(h3)
axs[3].plot(h4)
plt.show()









# # p12a = np.zeros(12)
# # p13a = np.zeros(12)
# # p34a = np.zeros(12)
# #
# # for i in range(1, 100):
# #     pd = get_pd(i - 50)
# #     p12a.index[i] = pd.index[1]
# #     p13a.index[i] = pd.index[2]
# #     p34a.index[i] = pd.index[3]
# #
# # plt.plot(p12a)
# # plt.plot(p13a)
# # plt.plot(p34a)
# # plt.show()
# #
# # wall = lambda x_w, y_w, z_w: x_w ** 2 / (150 ** 2) + y_w ** 2 / (100 ** 2) + z_w ** 2 / (38 ** 2)
#
# k = 0
#
# def get_pd(k):
#     t_len = 2.048 * 2
#     fs = 625000
#     sample = fs * t_len
#     ft = 30000
#     nl = -76  # noise level
#     h_dis = 11.5 / 304.8  # mm to ft
#     v_sound = 4858  # ft/s
#
#     # pinger location, first quadrant
#     px = 150 - math.sqrt(17 ** 2 - 14 ** 2) + k
#     py = 14
#     pz = -math.sqrt(38 ^ 2 * (1 - px ^ 2 / 150 ^ 2 - py ^ 2 / 100 ^ 2))
#
#     # hydrophone location, estimation, pointing pos_x direction
#     # bot in the top right quadrant in transdec, array mounted on the bot
#     # with pinger at the far end, bot just started off the starting deck
#     # if not sure about the array location simulated here, ask old members
#     hx1 = 150 / 3
#     hy1 = 100 / 4 * 3
#     hz1 = -2
#     hx2 = 150 / 3 - h_dis
#     hy2 = 100 / 4 * 3
#     hz2 = -2
#     hx3 = 150 / 3
#     hy3 = 100 / 4 * 3 + h_dis
#     hz3 = -2
#     hx4 = 150 / 3 - h_dis
#     hy4 = 100 / 4 * 3 + h_dis
#     hz4 = -2
#
#     # hx2 = 150/3 - sqrt(3)*h_dis/2;
#     # hy2 = 100/4*3 + h_dis/2;
#     # hz2 = -2;
#     # hx3 = 150/3;
#     # hy3 = 100/4*3 + h_dis;
#     # hz3 = -2;
#     # hx4 = 150/3 - sqrt(3)*h_dis/4;
#     # hy4 = 100/4*3 + h_dis/2;
#     # hz4 = -2 - h_dis/4*3;
#
#     dis1 = dis(px, py, pz, hx1, hy1, hz1)
#     dis2 = dis(px, py, pz, hx2, hy2, hz2)
#     dis3 = dis(px, py, pz, hx3, hy3, hz3)
#     dis4 = dis(px, py, pz, hx4, hy4, hz4)
#
#     # assume pinger ping at t = 0 for 4 ms, sound arrives at hydrophone with
#     # amplitude v = 5V
#     ping1 = np.zeros(sample)
#     ping2 = np.zeros(sample)
#     ping3 = np.zeros(sample)
#     ping4 = np.zeros(sample)
#
#     p1s = math.ceil(dis1 / v_sound * fs) + 13000 * 3
#     p2s = math.ceil(dis2 / v_sound * fs) + 13000 * 3
#     p3s = math.ceil(dis2 / v_sound * fs) + 13000 * 3
#     p4s = math.ceil(dis2 / v_sound * fs) + 13000 * 3
#
#     p12 = p1s - p2s
#     p13 = p1s - p3s
#     p34 = p3s - p4s
#     pd = [p12, p13, p34]
#
#     off1 = p1s - dis1 / v_sound * fs + 13000 * 3
#     off2 = p2s - dis2 / v_sound * fs + 13000 * 3
#     off3 = p3s - dis3 / v_sound * fs + 13000 * 3
#     off4 = p4s - dis4 / v_sound * fs + 13000 * 3
#
#     s = range(1, 0.004 * fs)
#     pingoff1 = 0.1 * np.sin((s + off1) / fs * ft * 2 * math.pi)  # complex conjugate transpose (np.matrix.H)
#     pingoff2 = 0.1 * np.sin((s + off2) / fs * ft * 2 * math.pi)
#     pingoff3 = 0.1 * np.sin((s + off3) / fs * ft * 2 * math.pi)
#     pingoff4 = 0.1 * np.sin((s + off4) / fs * ft * 2 * math.pi)
#
#     fig, axs = plt.subplots(4)
#     fig.subtitle('Sine Waves')
#     axs[0].plot(pingoff1)
#     axs[1].plot(pingoff2)
#     axs[2].plot(pingoff3)
#     axs[3].plot(pingoff4)
#
#     # need to figure out colons
#     # need to convert db
#
#     # noise, simulate from data, power shifts from -60db to -73db
#     mean = (-60 + nl) / 2
#     std = abs((-60 - nl) / 6)
#
#     h1 = ping1 + np.random.normal(mean, std, sample);
#     h2 = ping2 + np.random.normal(mean, std, sample);
#     h3 = ping3 + np.random.normal(mean, std, sample);
#     h4 = ping4 + np.random.normal(mean, std, sample);
#
#
# def dis(x, y, z, x1, y1, z1):
#     return math.sqrt((x - x1) ** 2 + (y - y1) ** 2 + (z - z1) ** 2)
#
#
# def db_to_intensity(i):
#     return i / (10)
#
#
# if __name__ == "__main__":
#
#     get_pd(k)
