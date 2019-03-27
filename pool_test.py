import sys
import subprocess
import os.path

if __name__ == "__main__":

    output_path = "/home/estellehe/Desktop/output/"

    freq = sys.argv[1]
    x = sys.argv[2]
    y = sys.argv[3]
    z = sys.argv[4]

    file_name = "625k_"+freq+"k_"+x+"_"+y+"_"+z+".csv"

    if os.path.isfile(output_path+file_name):
        file_name = file_name.replace(".csv", "(2).csv")


    subprocess.Popen(["python3", "/home/estellehe/Desktop/Acoustics/saleae_sampling.py", output_path+file_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
