import saleae
import subprocess
import time
import sys
import os.path
import paramiko

ip = '192.168.1.1'
username = 'robot'
pw = 'boebot'
x86_path = "/Users/estellehe/Documents/Robo/dummy.py"
arm_path = "/home/robot/Documents/dummy.py"
# 2 = 1250 kS/s, 3 = 625 kS/s, 4 = 125 kS/s
# sampling_rate = 3
#
# output_path = "/home/robot/Documents/output/"
#
if __name__ == "__main__":
#     try:
#         s = saleae.Saleae()
#     except:
#         subprocess.Popen(["/home/robot/Logic/Logic", "-socket"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#         time.sleep(15)
#         s = saleae.Saleae()
#         print("saleae software down, open saleae software before next script run")
#
#     #todo: add a while loop here to keep trying on get_active_device, timeout after 10s
#     try:
#         if s.get_active_device().active:
#             s.set_active_channels([], [0, 1, 2, 3])
#             s.set_capture_seconds(4.5)
#             s.set_sample_rate(s.get_all_sample_rates()[sampling_rate])
#     except:
#         exit()
#
#     fn = False
#     fn = input("filename: x y z version: ").split(' ')
#
#
#
#     while(fn):
#         file_name = "625k_40k_"+fn[0]+"_"+fn[1]+"_"+fn[2]+"("+fn[3]+").csv"
#         fn = False
#         s.capture_start_and_wait_until_finished()
#         s.export_data2(os.path.join(output_path, file_name), analog_channels=[0, 1, 2, 3])
#
#         time.sleep(12)
#         print("finish sampling")
#         fn = input("filename: x y z version: ").split(' ')

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(hostname=ip, port=22, username=username, password=pw)
    except Exception as e:
        print(e)
    sftp = client.open_sftp()
    sftp.put(x86_path, arm_path)
    sftp.close()
