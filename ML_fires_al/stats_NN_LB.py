import os
from os import listdir
from os.path import isfile, join
import pandas as pd

os.chdir('/home/sgirtsou/Documents/ML-dataset_newLU/csvs_withfire_results/LB_results')
mypath = '/home/sgirtsou/Documents/ML-dataset_newLU/csvs_withfire_results/LB_results'

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) and 'comb' in f]

combined_csv = pd.concat([pd.read_csv(f) for f in onlyfiles])

i = 1