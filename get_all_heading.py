import subprocess
import os


if __name__ == "__main__":
    root = "/home/estellehe/Desktop/Data"
    for file in (os.listdir(root)):
        if not file.startswith("out"):
            continue
        print file
        process = subprocess.Popen(["python", "cross_corr.py", os.path.join(root, file)], stdout = subprocess.PIPE)
        stddata, stderror = process.communicate()
        print stddata
