import os
import pandas
import threading
import time

df = pandas.DataFrame()
path = "/tmp/dummy.csv"

if __name__ == "__main__":
    df = pandas.read_csv(temp_path, skiprows=[1], skipinitialspace=True)
    print(df.to_string())
