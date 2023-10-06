import re
import os
import xarray as xr
import traceback
from parallel_file_process import par_files, walkmonthdays
import time
import multiprocessing as mp

def preprocess(df, dropfilters=None, fillnafilters=None, renames=None, horizfilters=None, \
                  addcols=None, calibfilters=None):
    if fillnafilters is not None:
        fillnacols=[c for c in df.columns if any(re.search(p,c) for p in fillnafilters)]
        for c in fillnacols:
            df[c].fillna(0, inplace=True)
    #print('before drop %d',len(df))
    #print('after drop %d', len(df))
    if dropfilters is not None:
        dropcols=[c for c in df.columns if any(re.search(p,c) for p in dropfilters)]
        df.drop(columns=dropcols,inplace=True)
        #print('dropped cols %s, col num before: %d, col num after: %d'%(dropcols,colnb, len(df.columns)))
        #print('dropped cols %s, col num before: %d, col num after: %d'%(dropcols,colnb, len(df.columns)))
    if horizfilters is not None:
        for hf in horizfilters:
            cond=eval(hf)
            df=df.copy()[cond]
    df.dropna(inplace=True)
    if addcols is not None:
        for addc in addcols:
            df[addc]=addcols[addc]
    if renames is not None:
        df.rename(columns=renames,inplace=True)
    if calibfilters is not None:
        for calib in calibfilters:
            '''v is for the eval expression use'''
            v=df[calib]
            df[calib]=eval(calibfilters[calib])
            #df.drop(columns=[calib], inplace=True)
            #df.rename(columns={calib+'_temp':calib}, inplace=True)
    return df


'''
convert netcdf to tabular
'''
def netcdf_to_csv(ncname, statname, tfolder, \
                  dropfilters=None, fillnafilters=None, renames=None, horizfilters=None, \
                  addcols=None, calibfilters=None
                  ):
    bname = os.path.basename(ncname)
    g1 = re.search('^(.*?)\.nc', bname)
    csvname = os.path.join(tfolder, g1.group(1) + '.csv')
    firedate=re.search('^(.*?)_df\.nc', bname).group(1)
    if os.path.isfile(csvname):
        print('Found ready Unormalized CSV')
        return csvname
    try:
        print('Converting %s'%ncname)
        ds=xr.open_dataset(ncname)
        ds_stat=xr.open_dataset(statname)
        dsdayall=xr.merge([ds,ds_stat],combine_attrs='drop')
        dfday=dsdayall.to_dataframe().reset_index()
        if type(addcols)==dict:
            addcols=dict({'firedate':firedate},**addcols)
        else:
            addcols={'firedate':firedate}
        dfday = preprocess(dfday, dropfilters, fillnafilters, renames, horizfilters, \
                           addcols, calibfilters)
        dfday.to_csv(csvname, index=False)
        print('Done Converting %s' % csvname)
        return csvname
    except:
        print('Failed to create unnormalized csv %s\n'%csvname+traceback.format_exc())
        return None


statname='/mnt/nvme2tb/ffp/datasets/images/static_aft_15.nc'
targetfolder='/mnt/nvme2tb/ffp/datasets/test/2020'
start=time.time()
print("Start Preprocessing and Converting netcdf to csv")
dayfiles=walkmonthdays('/mnt/nvme2tb/ffp/datasets/test/2020/final_dataset', '*nc','list')
proctime=par_files(netcdf_to_csv, sorted(dayfiles), mp.cpu_count()-2,
                   [statname, targetfolder,
                    ['curvature','index'], #dropfilters
                    [r'corine_(\d+)'], #fillnafilters
                    {'tp': 'rain_7_days', 'time':'firedate'}, #renames
                     None, # horizfilters
                     None, #addcols
                     {'firedate': 'v.str.replace("-","")'} # calibfilters
                    ]
                   )
dur=time.time()-start
print("Done in %d min and %d secs"%(int(dur/60), dur%60))