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

def par_norm_files(func, days, pthreads, args):
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

def normalizeset(csvfile, dropfilters=None):
    f=csvfile
    fnorm = os.path.join(os.path.dirname(f), os.path.basename(f).split('.')[0] + '_norm.csv')
    if f[-8:]=='norm.csv' or os.path.isfile(fnorm):
        return
    print('Start processing file %s'%f)
    df=pd.read_csv(f)
    #df=df.sample(frac=1)
    df.reset_index(inplace=True)
    #print('before drop %d',len(df))
    df=df.dropna()
    #print('after drop %d', len(df))
    if dropfilters is not None:
        colnb=len(df.columns)
        dropcols=[c for c in df.columns if any(re.search(p,c) for p in dropfilters)]
        df.drop(columns=dropcols,inplace=True)
        #print('dropped cols %s, col num before: %d, col num after: %d'%(dropcols,colnb, len(df.columns)))
    normdf = normdataset.normalize_dataset(df,aggrfile='/mnt/nvme2tb/ffp/datasets/norm_values_ref_final.json', check=False)
    normdf['firedate']=os.path.basename(f)[0:8]
    normdf.to_csv(fnorm,index=False)
    print('Done processing file %s' % f)

start=time.time()
print("Starting Normalization")
dayfiles=walkmonthdays('/mnt/nvme2tb/ffp/datasets/test/', '*_df.csv')
proctime=par_norm_files(normalizeset, dayfiles, mp.cpu_count()-2, [[r'corine_(\d+)','x\.1','y\.1']])
dur=time.time()-start
print("Done in %d min and %d secs"%(int(dur/60), dur%60))


#normalizeset(['/mnt/nvme2tb/ffp/datasets/test/20190601_df.csv'],[r'corine_(\d+)','x\.1','y\.1'])