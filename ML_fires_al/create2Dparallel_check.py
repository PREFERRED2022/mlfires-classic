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
import multiprocessing
import random
from functools import partial
from multiprocessing import Pool, Array, Process, Manager
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

def call_creategrid(fday):
    creategrid_xs(fday, rdiff, firstid, gridwidth, gridheight)

def printduration(start, mess=''):
    duration = time.time() - start
    start = time.time()
    print("Duration %s %.1f" % (mess,duration))
    return start

def fill_grid_from_dict_np(_id, ggrid, firstid, iddict, rdiff, field="id"):
    print(_id)
    try:
        row,col = get_grid_xy(_id, firstid, rdiff)
        print(row,col)
        print(ggrid)
        if field=="id":
            ggrid[row,col]=_id
        elif field=="array":
            ggrid[row,col]=iddict[_id]
    except:
        print('dict: %s'%iddict[_id])

def fill_grid_from_dict_np(_id, ggrid, firstid, iddict, rdiff, field="id"):
    print(_id)
    try:
        row,col = get_grid_xy(_id, firstid, rdiff)
        print(row,col)
        print(ggrid)
        if field=="id":
            ggrid[row,col]=_id
        elif field=="array":
            ggrid[row,col]=iddict[_id]
    except:
        print('dict: %s'%iddict[_id])


def get_grid_xy(firstid, rdiff, _id,):
    row =int((_id-firstid)/rdiff)
    col = int(_id-firstid-rdiff*row)
    return row,col

def creategrid_np(fday, exfeat):
    rdiff, firstid, gridwidth, gridheight = gridinfo()
    start = time.time()
    #daydict, featcolumns = loaddaydict(fday, exfeat,100)
    #featn=len(featcolumns)
    start = printduration(start, "of loading day:")
    ggrid_id = np.zeros((gridwidth, gridheight))
    #ggrid_patchpool = np.zeros((gridheight, gridwidth, featn))
    #idsnp = np.array(list(daydict.keys()))
    fill_grid_from_dict_v=np.vectorize(fill_grid_from_dict_np, excluded=["ggrid", "firstid", "iddict", "rdiff", "field"] )
    print(type(ggrid_id))
    fill_grid_from_dict_v(idsnp, ggrid_id, firstid, daydict, rdiff)
    start = printduration(start, "of filling grid ids:")
    #fill_grid_from_dict(ggrid_patchpool, firstid, daydict, rdiff, "array")
    #start = printduration(start, "of filling grid features:")
    return ggrid_id, ggrid_patchpool


def creategrid_xa(rdiff, firstid, gridwidth, gridheight, fday):
    #fday = '/data2/ffp/datasets/daily/2021/08/20210803_norm.csv'
    dt_df = dt.fread(fday)
    npday = dt_df[:, dt_df.names.index('id'):].to_numpy(dt.float32)

    #start = time.time()
    id2xy, grid = nppar.fillcube(7, npday, firstid, rdiff, gridwidth, gridheight)
    #end = time.time()
    #print(end - start)

    xaday = xarray.DataArray(data=grid, dims=["x", "y", "feature"],
                             coords=dict(x=range(gridwidth), y=range(gridheight), feature=range(len(dt_df.names) - 1)))

    orig_path=os.path.dirname(fday)
    fname = os.path.basename(fday)
    xaday.to_dataset().to_netcdf(os.path.join(orig_path,"%s_grid.nc"%(fname[0:8])))

def creategrid_xs(rdiff, firstid, gridwidth, gridheight, dayfile, cpus):
    #fday = '/data2/ffp/datasets/daily/2021/08/20210803_norm.csv'
    #print("processing day %s" % dayfile)
    try:
        orig_path = os.path.dirname(dayfile)
        fname = os.path.basename(dayfile)
        daygrid="%s_grid.nc" % (fname[0:8])
        #if os.path.isfile(os.path.join(orig_path, daygrid)): return
        dt_df = dt.fread(dayfile)
        firstfeat=dt_df.names.index('id')
        npday = dt_df[:, firstfeat:].to_numpy(dt.float32)

        #start = time.time()
        id2xy, grid = nppar.fillcube(cpus, npday, firstid, rdiff, gridwidth, gridheight)
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
        #print("Successfull convertion %s" % dayfile)
    except:
        print("Fail to convert %s" % dayfile)
        traceback.print_exc()
        with open("/data2/ffp/datasets/daily/failedgrids.log", "a") as f:
            f.write(dayfile)

def creategrid_xs_small(rdiff, firstid, gridwidth, gridheight, dayfile, pcpus, ccpus):
    # fday = '/data2/ffp/datasets/daily/2021/08/20210803_norm.csv'
    #print("processing day %s" % dayfile)
    try:
        orig_path = os.path.dirname(dayfile)
        fname = os.path.basename(dayfile)
        daygrid = "%s_grid.nc" % (fname[0:8])
        #if os.path.isfile(os.path.join(orig_path, daygrid)): return
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
        #print("Successfull convertion %s" % dayfile)
    except:
        print("Fail to convert %s" % dayfile)
        traceback.print_exc()
        with open("/data2/ffp/datasets/daily/failedgrids.log", "a") as f:
            f.write(dayfile)


def create_xs_files(creategrid, days, pcpus, ccpus):
    procs=[]
    dayscompleted=[]
    for cpu in range(pcpus):
        d=days.pop()
        dayscompleted+=[d]
        procs += [Process(target=creategrid, args=(d,pcpus,ccpus))]
        procs[cpu].start()
    while True:
        time.sleep(1)
        for p in procs:
            if not p.is_alive():
                procs.remove(p)
        if len(procs)<pcpus:
            if len(days)==0: break
            d = days.pop()
            dayscompleted += [d]
            procs += [Process(target=creategrid, args=(d,pcpus,ccpus))]
            procs[-1].start()

def assignrow(ggrid, tabrow):
    row, col = get_xy(tabrow[0])
    ggrid[row, col, :]=tabrow[:]

def assignrowshared(ggrid_sh, grid_shape, tabrows):
    for i in range(tabrows.shape[0]):
        tabrow=tabrows[i,:]
        try:
            row, col = get_xy(tabrow[0])
            idx = row*grid_shape[1]*grid_shape[2]+col*grid_shape[2]
            ggrid_sh[idx:idx+grid_shape[2]]=tabrow[:]
        except:
            "Error row: %s\n"%i+traceback.print_exc()

rdiff, firstid, gridwidth, gridheight = gridinfo()
creategrid = partial(creategrid_xs_small, rdiff, firstid, gridwidth, gridheight)
#dayfiles=walkmonthdays('/data2/ffp/datasets/daily/')
#with Pool(8) as p:
#    p.map(creategrid, dayfiles)

get_xy=partial(get_grid_xy, firstid, rdiff)

dayfiles=walkmonthdays('/data2/ffp/datasets/daily/')

'''
nruns=1
totalrun=0
for ccpus in range(5,16,5):
    for pcpus in range(5, 11, 5):
        start = time.time()
        create_xs_files(creategrid, dayfiles[:10], pcpus, ccpus)
        end = time.time()
        print('time: %.2f sec, python threads %s, cython threads %s' % (end - start, pcpus, ccpus))
'''
#fday='/data2/ffp/datasets/daily/2021/08/20210803_norm.csv'

fday='/data2/ffp/datasets/daily/2021/08/20210823_norm.csv'
dt_df = dt.fread(fday)
#creategrid_xs(rdiff, firstid, gridwidth, gridheight, fday)
firstfeat=dt_df.names.index('id')
npday = dt_df[:, firstfeat:].to_numpy(dt.float32)
#npday = dt_df[:, 1:].to_numpy(dt.float32)

maxcpus=multiprocessing.cpu_count()
featn = len(dt_df[:, firstfeat:].names)
ggrid = np.zeros((gridwidth, gridheight, featn))
ggrid[:,:]=np.nan
assignr=partial(assignrow, ggrid)
#assignr(list(npday)[0])
print('max cpu count %s'%maxcpus)
print('array rows: %s'%npday.shape[0])

nruns=1
totalrun=0
start = time.time()
for i in range(npday.shape[0]):
    assignr(npday[i])
end = time.time()
print('time: %.2f sec'%(end-start))
'''
nruns=1
totalrun=0
npdayl=list(npday)
for cpus in range(10,11):
    for run in range(nruns):
        start = time.time()
        with Pool(cpus) as p:
            p.map(assignr, npdayl)
        end = time.time()
        totalrun+=(end - start)
    print('average time: %.2f sec, cpus: %s'%(totalrun/nruns, cpus))
'''


#shDay[:] = np.nan
print('multiprocessing (with shared memory)')
nruns=1
totalrun=0
ncpus=10
for ncpus in range(1,17):
    start = time.time()
    procs=[]
    chunk = int(npday.shape[0] / ncpus)
    chunk_rem = npday.shape[0] % ncpus
    shDay = Array('f', gridwidth * gridheight * featn)
    for cpu in range(ncpus-1):
        procs += [Process(target=assignrowshared, args=(shDay, (gridwidth, gridheight, featn), npday[cpu*chunk:(cpu+1)*chunk]))]
        procs[cpu].start()
    procs += [Process(target=assignrowshared, args=(shDay, (gridwidth, gridheight, featn), npday[(ncpus-1)*chunk:]))]
    procs[-1].start()
    #while True:
    #    time.sleep(1)
    #    if all([not proc.is_alive() for proc in procs]):
    #        break
    for p in procs: p.join()
    shDay_np = np.frombuffer(shDay.get_obj(), dtype=np.float32).reshape((gridwidth, gridheight, featn))
    end = time.time()
    print('time: %.2f sec, cpu: %s'%((end-start),ncpus))
# Wrap X as an numpy array so we can easily manipulates its data.
# Copy data to our shared array.
#np.copyto(X_np, data)


'''
nruns=30
print('cpu changes, runs per experiment: %s'%nruns)
for nt in range(1,maxcpus+1):
    totalrun = 0
    for run in range(nruns):
        start = time.time()
        id2xy, grid = nppar.fillcube(nt, npday, firstid, rdiff, gridwidth, gridheight)
        end = time.time()
        totalrun+=(end - start)
    print('average time: %.2f sec, threads: %s'%((end - start),nt))

print('chunk changes')
nt = 10
print('rows / threads: %s threads: %s' % ((npday.shape[0] / nt),nt))
for cs in range(1,400000,10000):
    start = time.time()
    id2xy, grid = nppar.fillcube(nt, npday, firstid,  rdiff, gridwidth, gridheight, 'static', cs)
    end = time.time()
    print('time: %.2f chunk: %s'%((end - start), cs))
'''
print('schedule changes')
nt = 14
nruns=30
print('optimum rows / threads: %.0f, threads: %s, runs per experiment: %s' % ((npday.shape[0] / nt),nt, nruns))
for i in range(3):
    id2xy, grid = nppar.fillcube(nt, npday, firstid,  rdiff, gridwidth, gridheight, None)
schedules = [None, 'static', 'dynamic', 'guided']
random.shuffle(schedules)
for schedule in schedules:
    totalrun = 0
    for run in range(nruns):
        start = time.time()
        id2xy, grid = nppar.fillcube(nt, npday, firstid,  rdiff, gridwidth, gridheight, schedule)
        end = time.time()
        totalrun+=(end - start)
    print('average time: %.2f sec, schedule: %s'%(totalrun/nruns, schedule))


#xaday=xarray.DataArray(data=grid, dims=["x", "y", "feature"],  coords=dict(x=range(gridwidth), y=range(gridheight), feature=range(len(dt_df.names)-1)))
'''
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