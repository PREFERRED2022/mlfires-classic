import normdataset
import pandas as pd
import re
import multiprocessing as mp
import time
import os
import fileutils


'''
walk folders to find all dataset files
'''
def walkmonthdays(sfolder, pattern):
    dayfiles = []
    for dayf in fileutils.find_files(sfolder, pattern, listtype="walk"):
        dayfiles += [dayf]
    return dayfiles

def new_process(func, proclist, args):
    q = mp.Queue()
    proclist += [{'proc': mp.Process(target=func, args=args), 'queue': q}]
    proclist[-1]['proc'].start()

def par_files(func, days, pthreads, args):
    procs = []
    proctimetotal = 0
    dayscompleted = []
    #print(days)
    for cpu in range(pthreads):
        d = days.pop()
        dayscompleted += [d]
        #print('initial proc')
        new_process(func, procs, tuple([d]+args))
    while len(procs) > 0:
        time.sleep(0.1)
        for p in procs:
            try:
                proctimetotal += p['queue'].get_nowait()
            except:
                pass
            if not p['proc'].is_alive():
                #print('remove, tot procs: %d' % len(procs))
                procs.remove(p)
                #print('tot procs: %d' % len(procs))
        while len(procs) < pthreads:
            if len(days) == 0: break
            #print('new proc')
            d = days.pop()
            dayscompleted += [d]
            new_process(func, procs, tuple([d]+args))
    return proctimetotal

def normalizefile(csvfile, dropfilters=None, fillnafilters=None, renames=None, horizfilters=None, addcols=None):
    f=csvfile
    fnorm = os.path.join(os.path.dirname(f), os.path.basename(f).split('.')[0] + '_norm.csv')
    if f[-8:]=='norm.csv' or os.path.isfile(fnorm):
        return
    print('Start processing file %s'%f)
    df=pd.read_csv(f)
    #df=df.sample(frac=1)
    df.reset_index(inplace=True)
    if fillnafilters is not None:
        fillnacols=[c for c in df.columns if any(re.search(p,c) for p in fillnafilters)]
        for c in fillnacols:
            df[c].fillna(0, inplace=True)
    #print('before drop %d',len(df))
    df=df.dropna()
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
    if renames is not None:
        df.rename(columns=renames,inplace=True)
    if addcols is not None:
        for addc in addcols:
            df[addc]=0
    normdf = normdataset.normalize_dataset(df,aggrfile='/mnt/nvme2tb/ffp/datasets/norm_values_ref_perif.json', check=False)
    if 'firedate' not in normdf.columns:
        normdf['firedate']=os.path.basename(f)[0:8]
    normdf.to_csv(fnorm,index=False)
    print('Done processing file. Output %s' % fnorm)


start=time.time()
print("Starting Normalization")
#dayfiles=walkmonthdays('/mnt/nvme2tb/ffp/datasets/test/2019/', '*_df_sterea.csv')
dayfiles=walkmonthdays('/mnt/nvme2tb/ffp/datasets/test/2019/', '*_df.csv')
dayfiles=walkmonthdays('/mnt/nvme2tb/perifereia/', '2022*.csv')
#proctime=par_files(normalizefile, dayfiles, mp.cpu_count() - 2, [[r'corine_(\d+)', 'x\.1', 'y\.1']])
dur=time.time()-start
print("Done in %d min and %d secs"%(int(dur/60), dur%60))


#normalizefile('/mnt/nvme2tb/ffp/datasets/traindataset_new_attica.csv', [r'corine_(\d+)', 'x\.1', 'y\.1'])
#normalizefile('/mnt/nvme2tb/ffp/datasets/traindataset_new_sterea.csv', [r'corine_(\d+)', 'x\.1', 'y\.1'])

#normalizefile('/mnt/nvme2tb/ffp/datasets/train/train_sample_1_1_nof_att.csv', [r'corine_(\d+)', 'x\.1', 'y\.1'])
normalizefile('/mnt/nvme2tb/ffp/datasets/train/train_new_sample_1_2_attica.csv', None, [r'corine_(\d+)'])
