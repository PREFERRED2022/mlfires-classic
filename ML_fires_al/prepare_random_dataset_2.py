import pandas as pd
import normdataset
import check_and_prepare_dataset
from datetime import datetime
import numpy as np
import re
import csv
import bisect
import calendar
import fileutils

monthsetdir = '/home/aapos/Documents/newcrossval/'
olddataset = 'data/dataset_corine_level2_onehotenc.csv'
iddf = pd.read_csv(olddataset, usecols=['id','firedate'])#, dtype={'id': np.int32})
iddf['id'] = iddf['id'].astype(np.int64)
iddf = iddf.sort_values(by=['firedate'])
iddf['firedate']=iddf['firedate'].str.slice(0, 4)+iddf['firedate'].str.slice(5, 7)+iddf['firedate'].str.slice(8, 10)
dftrain = None
dfmissall = None
for year in range(2010,2020):
    for month in range (1,13):
        sdate = str(year)+'%02d'%month
        iddfmonth = iddf[iddf['firedate'].str.startswith(sdate)]
        if len(iddfmonth.index)==0:
            continue
        print('Collecting records for month %02d/%d'%(month,year))
        smonth = calendar.month_name[int(month)].lower()
        pattern = '*' + smonth + '*' + str(year) + '*_norm.csv'
        dftrainmonth = None
        cnt=0
        chunksize=500000
        for fmonth in fileutils.find_files(monthsetdir, pattern, listtype="walk"):
            print('reading file %s ...'%fmonth)
            for dfmonthpart in pd.read_csv(fmonth, chunksize=chunksize, dtype={'firedate': str, 'id': np.int64}):
                cnt+=1
                print('rows: %d'%(cnt*chunksize))
                dftrainpart = pd.merge(iddfmonth, dfmonthpart, on=['id', 'firedate'], how='inner')
                if len(dftrainpart.index > 0):
                    dftrainmonth = dftrainpart if dftrainmonth is None else pd.concat([dftrainmonth, dftrainpart])
                    break
        if dftrainmonth is None:
            dfmissmonth = iddfmonth
        else:
            dfmissmonth = pd.merge(iddfmonth, dftrainmonth, on=['id', 'firedate'], how='left', indicator=True)[['id','firedate','_merge']]
            dfmissmonth = dfmissmonth[dfmissmonth['_merge']=='left_only'][['id','firedate']]
        found = len(dftrainmonth.index) if dftrainmonth is not None else 0
        missed = len(dfmissmonth.index) if dfmissmonth is not None else 0
        print('Found %d records and missed %d for month %02d/%d'%(found, missed, month, year))
        dftrain = dftrainmonth if dftrain is None else pd.concat([dftrain, dftrainmonth])
        dfmissall = dfmissmonth if dfmissall is None else pd.concat([dfmissall, dfmissmonth])
if dftrain is not None:
    dftrain.to_csv('data/oldrandomnewfeat.csv',index=False)
if dfmissall is not None:
    dfmissall.to_csv('data/notfound_oldrandomnewfeat.csv',index=False)