import sys
sys.path.append("/mnt/nvme2tb/ffp/code/mlfires/ML_fires_al/")
import pathlib
import pandas as pd
import xarray as xr
import numpy as np
import os
import crop_dataset
import geopandas as gpd
import random

'''
Adds a hash string ID in the dataset from x y coords 
'''


def applyid(df):
    df['xposst'] = (df['x'] * 10000).apply('{:06.0f}'.format)
    df['yposst'] = (df['y'] * 10000).apply('{:06.0f}'.format)
    df['id'] = df['xposst'] + df['yposst']
    df.drop(columns=['xposst', 'yposst'], inplace=True)
    return df

def applyxy(df, x='x', y='y'):
    df[x] = df['id'].str.slice(0, 6).astype(int) / 10000
    df[y] = df['id'].str.slice(6, 12).astype(int) / 10000
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


def getcoarsedf(coarsen_c, dfday):
    ds = xr.open_dataset('/mnt/nvme2tb/ffp/datasets/images/20211030_df.nc')
    dsc = ds.coarsen(y=coarsen_c, boundary='trim').mean().coarsen(x=coarsen_c, boundary='trim').mean()
    coordtupar = dsc.stack(dim=['x', 'y']).dim.to_numpy()
    coordnp = np.array([*coordtupar])
    dfcoocrds = pd.DataFrame(coordnp, columns=['x', 'y'], dtype=float)
    dfcoocrds = applyid(dfcoocrds)
    coarsen_df = pd.merge(dfday, dfcoocrds, on=['id'], suffixes=("", "_c")).drop(columns=['x_c', 'y_c'])
    return coarsen_df


def extract_day(date, extented=True):
    csvfolder = '/mnt/nvme2tb/ffp/datasets/prod/'
    '''
    gdfperif = crop_dataset.getperif()
    crop_dataset.cropfile(os.path.join(csvfolder,date,'%s_norm.csv'%date),
                      os.path.join(csvfolder,date, gdfperif, '_greece'),
                      usexyid='id')
    '''
    if extented: ext='ext_'
    else: ext=''
    csvfile = os.path.join(csvfolder, date, '%s_norm_greece.csv' % date)

    # csv for xai input

    coarsen_coef = 31
    dfday = pd.read_csv(csvfile, dtype={'id': str})
    coarsedf = getcoarsedf(coarsen_coef, dfday)

    if extented:
        dfpred = extract_xy(pd.read_csv(os.path.join(csvfolder, date, "%s_pred_greece.csv" % date), dtype={'id': str}))
        dfexids, gdfbuffer = getextrapoints(coarsedf, dfpred, coarsen_coef)
        coarsedf = pd.merge(dfexids, dfday, on=['id'])

    xaifolder = '/mnt/nvme2tb/ffp/datasets/xai/%s' % date

    if not os.path.isdir(xaifolder): os.makedirs(xaifolder)
    csvcoarse = os.path.join(xaifolder, '%s_xai_%sinp.csv' % (date, ext))
    coarsedf.to_csv(csvcoarse, index=False)

    coarsedf=applyxy(coarsedf, 'xcoord', 'ycoord')
    coarsegdf=gpd.GeoDataFrame(coarsedf, geometry=gpd.points_from_xy(coarsedf['xcoord'], coarsedf['ycoord'], crs=4326)).\
        drop(columns=['xcoord', 'ycoord'])
    coarsegdf.to_file(os.path.join(xaifolder, '%s_xai_%spoints.shp' % (date, ext)))
    gdfbuffer.to_file(os.path.join(xaifolder, '%s_xai_%sbuffers.shp' % (date, ext)))
    return coarsedf


def extract_xy(dfxai):
    dfxai['x'] = dfxai['id'].str.slice(0, 6).astype(int) / 10000
    dfxai['y'] = dfxai['id'].str.slice(6, 12).astype(int) / 10000
    return dfxai


def getcenters(dfcoarse, dfpred):
    dfcenter = pd.merge(dfcoarse[['id', 'max_temp']], dfpred, on='id', how='right')
    dfcenter.loc[~dfcenter['max_temp'].isna(), 'max_temp'] = 1
    dfcenter.loc[dfcenter['max_temp'].isna(), 'max_temp'] = 0
    dfcenter.rename(columns={'max_temp': 'center'}, inplace=True)
    dfcenter['center'] = dfcenter['center'].astype(int)

    geom = gpd.points_from_xy(dfcenter['x'], dfcenter['y'], crs=4326)
    gdfcenter = gpd.GeoDataFrame(dfcenter, geometry=geom)

    return dfcenter


def getextrapoints(dfcoarse, dfpred, coarsen_coef):
    # merge the points from the coarsening with the rest of the dataset.
    # Create "center" column to mark which points are the chosen after the coarsening
    dfcenter = pd.merge(dfcoarse[['id', 'max_temp']], dfpred, on='id', how='right')
    dfcenter.loc[~dfcenter['max_temp'].isna(), 'max_temp'] = 1
    dfcenter.loc[dfcenter['max_temp'].isna(), 'max_temp'] = 0
    dfcenter.rename(columns={'max_temp': 'center'}, inplace=True)
    dfcenter['center'] = dfcenter['center'].astype(int)

    # create geodataframe with point geometries. Change to crs 2100 or 3857 for meter coordinates
    geom = gpd.points_from_xy(dfcenter['x'], dfcenter['y'], crs=4326)
    gdfcenter = gpd.GeoDataFrame(dfcenter, geometry=geom)
    gdfcenter = gdfcenter.to_crs(crs=3857)

    # spatial join of center points with all points around centers
    # using the coarsen coefficient to create a square buffer of 500*(coarsen_coef-1)/2 meters.
    # Needs a correction coefficient 1.18
    gdfcenter2 = gdfcenter.loc[gdfcenter['center'] == 1].copy()
    gdfcenter2['geometry'] = gdfcenter2.geometry.buffer((coarsen_coef - 1) / 2 * 500 * 1.18, cap_style=3)
    # gdfcenter2['geometry']=gdfcenter2.geometry.buffer((coarsen_coef-1)/2*0.075,cap_style=3)
    gdfcenter2.drop(columns=['ypred0', 'ypred1', 'x', 'y'], inplace=True)
    gdfsjoin = gdfcenter2.sjoin(gdfcenter, how="left")

    # find and select one point id for each risk level
    # for the points in the buffer keeping the initial center point
    sampleids = []
    for ind in gdfcenter2.index:
        sampleids += [gdfcenter2.loc[gdfcenter2.index == ind, 'id'].item()]
        for risk in range(1, 6):
            allrows = gdfsjoin.loc[(gdfsjoin.index == ind) \
                                   & (gdfsjoin['id_left'] != gdfsjoin['id_right']) \
                                   & (gdfsjoin['risk_left'] != gdfsjoin['risk_right']) \
                                   & (gdfsjoin['risk_right'] == risk)]
            if not allrows.empty:
                # print(ind,allrows.iloc[0]["id_right"])
                celln = random.randint(0, len(allrows) - 1)
                # print(len(allrows), celln)
                sampleids += [allrows.iloc[celln]["id_right"]]
    dfextids = pd.DataFrame(sampleids, columns=['id'])
    return dfextids, gdfcenter2

#extract_day('20230825')

for d in range(25,26):
    date='202308'+str(d)
    print('Processing day %s'%date)
    extract_day(date)