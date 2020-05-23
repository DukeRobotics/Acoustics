import saleae
import subprocess
import time

# 2 = 1250 kS/s, 3 = 625 kS/s, 4 = 125 kS/s
rates = {2: 1250000, 3: 625000, 4: 125000}
sampling_rate = 3
fs = rates[sampling_rate]
pingc = fs*0.004

output_path = "/home/estellehe/Desktop/output/"

if __name__ == "__main__":
    try:
        s = saleae.Saleae()
    except:
        subprocess.Popen(["/home/estellehe/Desktop/Logic/Logic"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(15)
        s = saleae.Saleae()
        print("saleae software down, open saleae software before next script run")

    try:
        if s.get_active_device().active:
            s.set_active_channels([], [0, 1, 2, 3])
            s.set_capture_seconds(3)
            s.set_sample_rate(s.get_all_sample_rates()[sampling_rate])
    except:
        exit()

    fn = input("filename: x y z version: ").split(' ')
    while fn:
        file_name = "625k_40k_"+fn[0]+"_"+fn[1]+"_"+fn[2]+"("+fn[3]+").csv"
        s.capture_start_and_wait_until_finished()
        s.export_data2(output_path+file_name, analog_channels=[0, 1, 2, 3])
        time.sleep(10)
        print("finish sampling")
        fn = input("filename: x y z version: ").split(' ')
