import sys
sys.path.append("/mnt/nvme2tb/ffp/code/mlfires/ML_fires_al/")
import pathlib
import pandas as pd
import xarray as xr
import numpy as np
import os
import crop_dataset

'''
Adds a hash string ID in the dataset from x y coords 
'''
def applyid(df):
    df['xposst'] = (df['x'] * 10000).apply('{:06.0f}'.format)
    df['yposst'] = (df['y'] * 10000).apply('{:06.0f}'.format)
    df['id'] = df['xposst'] + df['yposst']
    df.drop(columns=['xposst', 'yposst'], inplace=True)
    return df

'''
Creates coarsen dataset subgrid from the initial dataset.


Input 
    coarsen_c : The number of points in the subgrid is 1/( coarsen_c * coarsen_c) 
                We need to use odd numbers to maintain existing points coordinates in the subgrid.
    file : The csv file to merge

Output  
    Corsen tabular pandas dataframe that contains only the instances of the points in the grid
'''

def getcoarsedf(coarsen_c, file):
    ds = xr.open_dataset('/mnt/nvme2tb/ffp/datasets/images/20211030_df.nc')
    dsc = ds.coarsen(y=coarsen_c, boundary='trim').mean().coarsen(x=coarsen_c, boundary='trim').mean()
    coordtupar = dsc.stack(dim=['x', 'y']).dim.to_numpy()
    coordnp = np.array([*coordtupar])
    dfcoocrds = pd.DataFrame(coordnp, columns=['x', 'y'], dtype=float)
    dfcoocrds = applyid(dfcoocrds)
    dfday = pd.read_csv(file, dtype={'id': str})
    coarsen_df = pd.merge(dfday, dfcoocrds, on=['id'], suffixes=("", "_c")).drop(columns=['x_c', 'y_c'])
    return coarsen_df

def extract_day(date):
    csvfolder = '/mnt/nvme2tb/ffp/datasets/prod/'
    '''
    gdfperif = crop_dataset.getperif()
    crop_dataset.cropfile(os.path.join(csvfolder,date,'%s_norm.csv'%date),
                      os.path.join(csvfolder,date, gdfperif, '_greece'),
                      usexyid='id')
    '''

    csvfile = os.path.join(csvfolder, date, '%s_norm_greece.csv' % date)

    # csv for xai input
    coarsedf = getcoarsedf(31, csvfile)
    xaifolder = '/mnt/nvme2tb/ffp/datasets/xai/%s' % date
    if not os.path.isdir(xaifolder): os.makedirs(xaifolder)
    csvcoarse = os.path.join(xaifolder, '%s_xai_inp.csv' % date)
    coarsedf.to_csv(csvcoarse, index=False)


for d in range(25,29):
    print('Explaining day %s' % d)
    date='202308'+str(d)
    extract_day(date)
