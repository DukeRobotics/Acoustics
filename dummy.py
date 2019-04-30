import os
import pandas
import threading
import time

df = pandas.DataFrame()
temp_path = "/tmp/dummy.csv"

def pandas_read():
    global df
    df = pandas.read_csv(temp_path, skiprows=[1], skipinitialspace=True)
    return

if __name__ == "__main__":
    # read = threading.Thread(target=pandas_read)
    # read.start()
    # read.join()
    df = pandas.read_csv(temp_path, skiprows=[1], skipinitialspace=True)
    print(df.to_string())
