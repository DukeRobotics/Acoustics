import saleae
import subprocess
import time

# 2 = 1250 kS/s, 3 = 625 kS/s, 4 = 125 kS/s
sampling_rate = 3

if __name__ == "__main__":

    try:
        s = saleae.Saleae()
    except:
        subprocess.Popen(["/home/estellehe/Desktop/Logic/Logic"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(15)
        s = saleae.Saleae()
        print("saleae software down, open saleae software before next script run")
        #print("software not opened or no device detected")
        #exit()

    #todo: add a while loop here to keep trying on get_active_device, timeout after 10s
    try:
        if s.get_active_device().active:
            s.set_active_channels([], [0, 1, 2, 3])
            s.set_capture_seconds(1)
            s.set_sample_rate(s.get_all_sample_rates()[sampling_rate])
    except:
        exit()

    s.capture_start_and_wait_until_finished()

    # change to a series of csv
    try:
        file = open("saleae_data.csv", 'r')
    except IOError:
        file = open("saleae_data.csv", 'w')

    #todo: add a while loop to wait till process is finished and then export data
    if s.is_processing_complete():
        s.export_data2("/home/estellehe/Desktop/Acoustics/saleae_data.csv", analog_channels=[0, 1, 2, 3])
