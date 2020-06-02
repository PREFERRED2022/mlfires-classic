import os
from os import listdir
from os.path import isfile, join
import pandas as pd

mypath = '/home/sgirtsou/Documents/ML-dataset_newLU/csvs_withfire'
os.chdir('/home/sgirtsou/Documents/ML-dataset_newLU/csvs_withfire')

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for file in onlyfiles:
    if file.endswith('csv'):
        df_greece = pd.read_csv(file)
        df_results =df_greece[0:1]
        df_results.to_csv('/home/sgirtsou/Documents/ML-dataset_newLU/csvs_withfire_results/' + file[0:12] + '_res.csv')