import saleae
import subprocess
import time
import sys
import os.path

# 2 = 1250 kS/s, 3 = 625 kS/s, 4 = 125 kS/s
sampling_rate = 3
fs = 625000
pingc = pingc = fs*0.004

output_path = "/home/robot/Documents/output/"

def moving_average_max(a, n = pingc) :
    weights = np.repeat(1.0, n)/n
    alist = np.convolve(a, weights, 'valid')
    maxa = 0
    maxi = 0
    for k in range(len(alist)):
    	ave = alist[k]
    	if ave > maxa:
    	    maxa = ave
    	    maxi = k
    return maxi

if __name__ == "__main__":
    try:
        s = saleae.Saleae()
    except:
        subprocess.Popen(["/home/robot/Logic/Logic", "-socket"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(15)
        s = saleae.Saleae()
        print("saleae software down, open saleae software before next script run")
        #print("software not opened or no device detected")
        #exit()

    #todo: add a while loop here to keep trying on get_active_device, timeout after 10s
    try:
        if s.get_active_device().active:
            s.set_active_channels([], [0, 1, 2, 3])
            s.set_capture_seconds(4.5)
            s.set_sample_rate(s.get_all_sample_rates()[sampling_rate])
    except:
        exit()

    # try:
    #     file = open(sys.argv[1], 'r')
    # except IOError:
    #     file = open(sys.argv[1], 'w')
    fn = False
    fn = input("filename: x y z version: ").split(' ')

#     file_name = "625k_40k_"+fn[0]+"_"+fn[1]+"_"+fn[2]+"("+fn[3]+").csv"
#     s.capture_start_and_wait_until_finished()
#     s.export_data2(os.path.join(output_path, file_name), analog_channels=[0, 1, 2, 3])
# #
# # check how long does 4.5 second sampling need
#     time.sleep(12)
#     print("finish sampling")
#
#     p = subprocess.Popen(["python3", "/home/robot/Documents/Acoustics/cross_corr_4.py", os.path.join(output_path, file_name)], stdout=subprocess.PIPE)
#
#     while p.poll() is None:
#         l = p.stdout.readline() # This blocks until it receives a newline.
#         print(l)
#
#     print(p.stdout.read())



    while(fn):
        file_name = "625k_40k_"+fn[0]+"_"+fn[1]+"_"+fn[2]+"("+fn[3]+").csv"
        fn = False
        s.capture_start_and_wait_until_finished()
        s.export_data2(os.path.join(output_path, file_name), analog_channels=[0, 1, 2, 3])

        time.sleep(12)
        print("finish sampling")
        fn = input("filename: x y z version: ").split(' ')
