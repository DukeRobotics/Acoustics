import sys
import subprocess
import os.path
import pty

if __name__ == "__main__":
    #channel 1 is left bottom corner
    output_path = "/home/estellehe/Desktop/output/"

    freq = "40"
    x = sys.argv[1]
    y = sys.argv[2]
    z = sys.argv[3]

    file_name = "625k_"+freq+"k_"+x+"_"+y+"_"+z+".csv"

    if os.path.isfile(output_path+file_name):
        file_name = file_name.replace(".csv", "(2).csv")

    out_r, out_w = pty.openpty()
    process = subprocess.Popen(["python3", "/home/estellehe/Desktop/Acoustics/saleae_sampling.py", file_name], stdout = out_w)
    os.close(out_w) # if we do not write to process, close this.
    while True:
        try:
            output = os.read(out_r, 1000).decode()
        except OSError as e:
            # if e.errno != errno.EIO: raise
            output = ""
        if not output:
            break
        else:
            print(output)
