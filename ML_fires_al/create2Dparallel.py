#import prange_test
import nppar
import time
import numpy as np
import math
import datatable as dt
import time
import xarray
import os
import fileutils
import traceback
from datetime import datetime
from multiprocessing import Pool, Queue, Process
from functools import partial

def gridinfo():
    rdiff=2227
    minnorth=333237
    #maxwest=1160624
    minwest=1156167
    maxeast=2747676
    maxsouth=3504175
    gridwidth=((maxeast-minwest) % rdiff) + 1
    firstid = minwest-math.ceil((minwest-minnorth) / rdiff)*rdiff
    gridheight=math.ceil((maxsouth-firstid) / rdiff)
    return rdiff, firstid, gridwidth, gridheight

def walkmonthdays(sfolder):
    #sfolder = '/data2/ffp/datasets/daily/2015/08'
    exfeat = ["id", "firedate"]
    dayfiles=[]
    for dayf in fileutils.find_files(sfolder, '*_norm.csv', listtype="walk"):
        dayfiles+=[dayf]
        #print(fday)
        '''
        try:
            #fday = '/data2/ffp/datasets/daily/2021/08/20210804_norm.csv'
            creategrid_xs(fday, rdiff, firstid, gridwidth, gridheight)
        except:
            print("Fail to convert %s"%fday)
            traceback.print_exc()
        '''
    return dayfiles

def creategrid_xs(rdiff, firstid, gridwidth, gridheight, dayfile):
    #fday = '/data2/ffp/datasets/daily/2021/08/20210803_norm.csv'
    print("processing day %s" % dayfile)
    try:
        orig_path = os.path.dirname(dayfile)
        fname = os.path.basename(dayfile)
        daygrid="%s_grid.nc" % (fname[0:8])
        #if os.path.isfile(os.path.join(orig_path, daygrid)): return
        dt_df = dt.fread(dayfile)
        firstfeat=dt_df.names.index('id')
        npday = dt_df[:, firstfeat:].to_numpy(dt.float32)

        #start = time.time()
        id2xy, grid = nppar.fillcube(7, npday, firstid, rdiff, gridwidth, gridheight)
        #end = time.time()
        #print(end - start)

        vardict = {}
        for i in range(firstfeat, len(dt_df.names)):
            varname = dt_df.names[i]
            if dt_df.names[i] == 'x' or dt_df.names[i] == 'y':
                varname = '%spos' % varname
            vardict[varname] = (["x", "y", "time"], np.expand_dims(grid[:, :, i-firstfeat], axis=2))

        t = datetime.strptime(os.path.basename(dayfile)[0:8], '%Y%m%d')
        xsday = xarray.Dataset(data_vars=vardict, coords=dict(x=range(gridwidth), y=range(gridheight), time=[t]))
        xsday.to_netcdf(os.path.join(orig_path, daygrid))
        print("Successfull convertion %s" % dayfile)
    except:
        print("Fail to convert %s" % dayfile)
        traceback.print_exc()
        with open("/data2/ffp/datasets/daily/failedgrids.log", "a") as f:
            f.write(dayfile)


def creategrid_xs_small(rdiff, firstid, gridwidth, gridheight, dayfile, pcpus, ccpus, queue):
    # fday = '/data2/ffp/datasets/daily/2021/08/20210803_norm.csv'
    print("processing day %s" % dayfile)
    try:
        stpr = time.process_time()
        orig_path = os.path.dirname(dayfile)
        fname = os.path.basename(dayfile)
        daygrid = "%s_grid.nc" % (fname[0:8])
        if os.path.isfile(os.path.join(orig_path, daygrid)):
            print('Done\n')
            return
        dt_df = dt.fread(dayfile, nthreads=1)#pcpus)
        firstfeat = dt_df.names.index('id')
        #npday = dt_df[:, firstfeat:].to_numpy(dt.float32)

        dynamic_feat=['id', 'max_temp', 'min_temp', 'mean_temp', 'res_max',
         'dom_vel', 'rain_7days', #'dem', 'slope', 'curvature', 'aspect',
         'ndvi_new', 'evi', 'lst_day', 'lst_night', 'max_dew_temp',
         'mean_dew_temp', 'min_dew_temp', 'fire', 'dir_max_1', 'dir_max_2',
         'dir_max_3', 'dir_max_4', 'dir_max_5', 'dir_max_6', 'dir_max_7',
         'dir_max_8', 'dom_dir_1', 'dom_dir_2', 'dom_dir_3', 'dom_dir_4',
         'dom_dir_5', 'dom_dir_6', 'dom_dir_7', 'dom_dir_8', #'corine_111',
        # 'corine_112', 'corine_121', 'corine_122', 'corine_123', 'corine_124',
        # 'corine_131', 'corine_132', 'corine_133', 'corine_141', 'corine_142',
        # 'corine_211', 'corine_212', 'corine_213', 'corine_221', 'corine_222',
        # 'corine_223', 'corine_231', 'corine_241', 'corine_242', 'corine_243',
        # 'corine_244', 'corine_311', 'corine_312', 'corine_313', 'corine_321',
        # 'corine_322', 'corine_323', 'corine_324', 'corine_331', 'corine_332',
        # 'corine_333', 'corine_334', 'corine_411', 'corine_412', 'corine_421',
        # 'corine_422', 'corine_511', 'corine_512', 'corine_521', 'wkd_0',
        # 'wkd_1', 'wkd_2', 'wkd_3', 'wkd_4', 'wkd_5', 'wkd_6', 'month_7',
        # 'month_4', 'month_5', 'month_6', 'month_8', 'month_9',
        'frequency','f81',]# 'xpos', 'ypos']

        dyn_df = dt_df[:, dynamic_feat]
        npday = dyn_df.to_numpy(dt.float32)

        # start = time.time()
        id2xy, grid = nppar.fillcube(ccpus, npday, firstid, rdiff, gridwidth, gridheight)
        # end = time.time()
        # print(end - start)

        vardict = {}
        for i in range(0, len(dyn_df.names)):
            varname = dyn_df.names[i]
            if dyn_df.names[i] == 'x' or dyn_df.names[i] == 'y':
                varname = '%spos' % varname
            vardict[varname] = (["x", "y", "time"], np.expand_dims(grid[:, :, i], axis=2))

        t = datetime.strptime(os.path.basename(dayfile)[0:8], '%Y%m%d')
        xsday = xarray.Dataset(data_vars=vardict, coords=dict(x=range(gridwidth), y=range(gridheight), time=[t]))
        xsday.to_netcdf(os.path.join(orig_path, daygrid))
        print("Successfull convertion %s" % dayfile)
        epr = time.process_time()
        queue.put(epr-stpr)
    except:
        print("Fail to convert %s" % dayfile)
        traceback.print_exc()
        with open("/data2/ffp/datasets/daily/failedgrids.log", "a") as f:
            f.write('%s %s'%(datetime.now(),dayfile))

def new_process(proclist, day, pthreads, cthreads):
    q=Queue()
    proclist += [{'proc':Process(target=creategrid, args=(day, pthreads, cthreads, q)), 'queue': q}]
    proclist[-1]['proc'].start()

def create_xs_files(creategrid, days, pthreads, cthreads):
    procs=[]
    proctimetotal=0
    dayscompleted=[]
    for cpu in range(pthreads):
        d=days.pop()
        dayscompleted+=[d]
        new_process(procs, d, pthreads, cthreads)
    while len(procs)>0:
        time.sleep(1)
        for p in procs:
            try:
                proctimetotal+=p['queue'].get_nowait()
            except:
                pass
            if not p['proc'].is_alive():
                procs.remove(p)
        while len(procs)<pthreads:
            if len(days)==0: break
            d = days.pop()
            dayscompleted += [d]
            new_process(procs, d, pthreads, cthreads)
    return proctimetotal

rdiff, firstid, gridwidth, gridheight = gridinfo()
creategrid = partial(creategrid_xs_small, rdiff, firstid, gridwidth, gridheight)
dayfiles=walkmonthdays('/data2/ffp/datasets/daily/')
#creategrid(dayfiles[0])
create_xs_files(creategrid, dayfiles, 10, 5)

'''
fday='/data2/ffp/datasets/daily/2021/08/20210803_norm.csv'
dt_df = dt.fread(fday)
npday = dt_df[:, 1:].to_numpy(dt.float32)


start = time.time()
id2xy, grid = nppar.fillcube(7, npday, firstid,  rdiff, gridwidth, gridheight)
end = time.time()
print(end - start)

#xaday=xarray.DataArray(data=grid, dims=["x", "y", "feature"],  coords=dict(x=range(gridwidth), y=range(gridheight), feature=range(len(dt_df.names)-1)))

t = datetime.strptime(os.path.basename(fday)[0:8], '%Y%m%d')

vardict={}
for i in range(1,len(dt_df.names)):
    varname=dt_df.names[i]
    if dt_df.names[i]=='x' or dt_df.names[i]=='y':
        varname='%spos'%varname
    vardict[varname]=(["x", "y", "time"], np.expand_dims(grid[:, :, i - 1], axis=2))

xsday=xarray.Dataset(data_vars=vardict, coords=dict(x=range(gridwidth), y=range(gridheight), time=[t]))
i=1
'''


#print('finished mydot: {} s'.format(time.clock()-t))

#print('Passed test:', np.allclose(c, c2))