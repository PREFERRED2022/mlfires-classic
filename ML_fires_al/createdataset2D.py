import os
import pandas as pd
import calendar
import time
import csv
import datatable as dt
import numpy as np
import math
import fileutils
import time
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import traceback
import xarray as xr
import prange_test

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

def get_grid_xy(_id, firstid, rdiff):
    row =int((_id-firstid)/rdiff)
    col = _id-firstid-rdiff*row
    return row,col

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


def fill_grid_from_dict_v2(_id, idx, grid2D, dtnp, firstid, rdiff):
    try:
        #print(_id, idx, grid2D, dtnp, firstid, rdiff)
        row,col = get_grid_xy(int(_id), firstid, rdiff)
        print(row,col)
        print(_id,idx)
        grid2D[row,col]=dtnp[idx,:].to_numpy()
    except:
        traceback.print_exc()

def fill_grid_from_dict(ggrid, firstid, iddict, rdiff, field="id"):
    cnt=0
    #print("Filling grid %s..."%field)
    for _id in iddict:
        try:
            row,col = get_grid_xy(_id, firstid, rdiff)
            if field=="id":
                ggrid[row,col]=_id
            elif field=="array":
                ggrid[row,col]=iddict[_id]
            #cnt+=1
            #if cnt % 50000 == 0: print(cnt)
        except:
            print('dict: %s'%iddict[_id])
            continue
    return ggrid

def loaddaydict(fday, exfeat, n=None):
    print("Loading %s..."%fday)
    featcolumns=getfeatcolumns(fday, exfeat)
    daydict={}
    with open(fday, 'r') as featfile:
        cnt=0
        #if not checksequence(fday, exfeat): return {}
        for dictline in csv.DictReader(featfile):
            id_int = int(float(dictline["id"]))
            for k in exfeat:
                if k in dictline:
                    del dictline[k]
            featvals=list(dictline.values())
            daydict[id_int]=np.array(featvals)
            cnt+=1
            if n and n>=cnt: break
    return daydict,featcolumns

def printduration(start, mess=''):
    duration = time.time() - start
    start = time.time()
    print("Duration %s %.1f" % (mess,duration))
    return start

def creategrid(fday, exfeat):
    rdiff, firstid, gridwidth, gridheight = gridinfo()
    start = time.time()
    daydict, featcolumns = loaddaydict(fday, exfeat)
    featn=len(featcolumns)
    start = printduration(start, "of loading day:")
    ggrid_id = np.zeros((gridwidth, gridheight))
    ggrid_patchpool = np.zeros((gridheight, gridwidth, featn))
    fill_grid_from_dict(ggrid_id, firstid, daydict,rdiff)
    start = printduration(start, "of filling grid ids:")
    fill_grid_from_dict(ggrid_patchpool, firstid, daydict, rdiff, "array")
    start = printduration(start, "of filling grid features:")
    return ggrid_id, ggrid_patchpool

def creategrid_np(fday, exfeat):
    rdiff, firstid, gridwidth, gridheight = gridinfo()
    start = time.time()
    daydict, featcolumns = loaddaydict(fday, exfeat,100)
    featn=len(featcolumns)
    start = printduration(start, "of loading day:")
    ggrid_id = np.zeros((gridwidth, gridheight))
    ggrid_patchpool = np.zeros((gridheight, gridwidth, featn))
    idsnp = np.array(list(daydict.keys()))
    fill_grid_from_dict_v=np.vectorize(fill_grid_from_dict_np, excluded=["ggrid", "firstid", "iddict", "rdiff", "field"] )
    print(type(ggrid_id))
    fill_grid_from_dict_v(idsnp, ggrid_id, firstid, daydict,rdiff)
    start = printduration(start, "of filling grid ids:")
    #fill_grid_from_dict(ggrid_patchpool, firstid, daydict, rdiff, "array")
    #start = printduration(start, "of filling grid features:")
    return ggrid_id, ggrid_patchpool

def createpatches(fday, exfeat, patchwidth=64, padding=0):
    ggrid_id, ggrid_patchpool = creategrid(fday, exfeat)
    rdiff, firstid, gridwidth, gridheight = gridinfo()
    pw=patchwidth
    patchlist=[]
    #print(np.count_nonzero(ggrid_patchpool))
    for w in range(0, gridwidth, pw):
        for h in range(0, gridheight, pw):
            patch=ggrid_patchpool[h:h+pw,w:w+pw,:]
            if np.count_nonzero(patch)>0:
                pass
            if np.count_nonzero(ggrid_patchpool[h:h+pw,w:w+pw,:])>0:
                pass
            patchlist.append(patch)
    #start = printduration(start, "of patch creation:")
    return patchlist

def walkmonthdays(sfolder):
    #sfolder = '/data2/ffp/datasets/daily/2015/08'
    exfeat = ["id", "firedate"]
    for fday in fileutils.find_files(sfolder, '*_norm.csv', listtype="walk"):
        print(fday)
        #patchlist=createpatches(fday, exfeat)
        ggrid_id, ggrid_patchpool = creategrid(fday, exfeat)
        #if np.count_nonzero(ggrid_patchpool[:,:,17])>0:
        #    print("day contains fire!!")
        #    break
        #else:
        #    print("no fire in day")
    return ggrid_id, ggrid_patchpool

def walkmonthdays2():
    sfolder = '/data2/ffp/datasets/daily/2021/08'
    exfeat = ["","id", "firedate"]
    fday='/data2/ffp/datasets/daily/2021/08/20210801_norm.csv'
    featcolumns = getfeatcolumns(fday, exfeat)
    featn = len(featcolumns)
    rdiff, firstid, gridwidth, gridheight = gridinfo()
    monthgrid = np.zeros((gridheight, gridwidth, featn*31), dtype=np.float16)
    i=0
    for fday in fileutils.find_files(sfolder, '*_norm.csv', listtype="walk"):
        dayf = os.path.basename(fday)
        print(dayf)
        #patchlist=createpatches(fday, exfeat)
        ggrid_id, ggrid_patchpool = creategrid(fday, exfeat)
        dint = int(dayf[6:8])-1
        monthgrid[:,:,dint*featn:(dint+1)*featn] = ggrid_patchpool.copy()
        if i==3: break
        i+=1
    return monthgrid

def getfeatcolumns(fday, exfeat):
    with open(fday) as f:
        reader = csv.reader(f)
        for featcolumns in reader:
            break
    for k in exfeat:
        if k in featcolumns:
            featcolumns.remove(k)
    featdict={k:featcolumns.index(k) for k in featcolumns }
    return featdict

def plotfeatture(feat, featdict, ggrid, cmap='Spectral'):
    gfeat=ggrid[:,:,featdict[feat]]
    gfeat[gfeat<0] = 0
    gfeat[gfeat>1] = 1
    ax = sns.heatmap(gfeat, cmap=cmap)
    plt.show()

def plotfeatures(n, m, start, ggrid, filt=[]):
    sbplt=(n,m)
    f, ax= plt.subplots(sbplt[0],sbplt[1], figsize=(m*4,n*3), sharey=True, sharex=True)
    cnt=0
    g={}
    for feat in sorted(featdict):
        if any(flt in feat for flt in filt): continue
        cnt+=1
        if cnt<start: continue
        xplt=(cnt-start)%sbplt[1]
        yplt=(cnt-start)//sbplt[1]
        if n>1:
            pltax=ax[yplt,xplt]
        else:
            pltax=ax[cnt-start]
        ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])
        print('Plotting %s feature : %s at subplot %d,%d'%(ordinal(featdict[feat]),feat,yplt,xplt))
        gfeat=ggrid[:,:,featdict[feat]].copy()
        l0 = gfeat[gfeat<0].size
        g1 = gfeat[gfeat>1].size
        if l0>0 or g1>0:
            print('Warning! %d values <0 and %d values >1 for feature %d:%s. Replacing with limit...'%(l0,g1,cnt,feat))
            gfeat[gfeat<0] = 0
            gfeat[gfeat>1] = 1
        g[cnt] = sns.heatmap(gfeat, cmap='Spectral_r', ax=pltax)
        g[cnt].set_title(feat)
        if cnt-start+1==n*m: break
    plt.show()

def featurestat(ggrid):
    featdict = getfeatcolumns(fday, exfeat)
    stats=[]
    cnt=0
    for feat in sorted(featdict):
        gfeat=ggrid[:,:,featdict[feat]]
        l0 = gfeat[gfeat<0].size
        g1 = gfeat[gfeat>1].size
        if l0>0 or g1>0:
            print('Warning! %d values <0 and %d values >1 for feature %d:%s'%(l0,g1,cnt,feat))
        stats+=[{'feature':feat, 'min':gfeat.min(), 'max':gfeat.max(), 'mean': gfeat.mean(), 'std': gfeat.std()}]
    statspd=pd.DataFrame(stats)
    return statspd

def foo(x, a):
    return x+a

#daydict,featcolumns = loaddaydict(fday, exfeat)
#fday='/data2/ffp/datasets/daily/2015/09/20150911_norm.csv'
#fday='/data2/ffp/datasets/daily/2015/08/20150818_norm.csv'
fday='/data2/ffp/datasets/daily/2020/07/20200723_norm.csv'
#fday='/data2/ffp/datasets/daily/2021/08/20210810_norm.csv'
exfeat = ["","id", "firedate"]

start = time.time()
dt_df = dt.fread(fday)
end = time.time()
print(end - start)

#dt_df[:, 'id'].to_numpy()


#dt_df[1, :]['id']

#dt_df[1, :].shape[1]

rdiff, firstid, cols, rows = gridinfo()
ggrid_id_patchpool_np=np.zeros((rows, cols,dt_df[1, :].shape[1]))
ggrid_id_np = np.zeros((rows, cols))

prange_test.npapplypar(1, foo, np.array([0,1,2,3]), 2)
i=0




#fill_grid_from_dict_v2(335462,1,ggrid_id_patchpool_np,dt_df,firstid, rdiff)
#ggrid_id_patchpool_np[1,1058]
'''
grid_ids_np=dt_df[:, 'id'].to_numpy()
grid_ids_da=da.from_array(grid_ids_np, chunks=(20000))
get_grid_xy_v=np.vectorize(get_grid_xy)
new_da=get_grid_xy_v(grid_ids_da,firstid,rdiff)
fill_grid_from_dict_v=np.vectorize(fill_grid_from_dict_v2)#(_id, grid2D, dtnp, firstid, rdiff)
fill_grid_from_dict_v.excluded.add(2)
fill_grid_from_dict_v(grid_ids_da, grid_idx_da, ggrid_id_patchpool_np, dt_df, firstid, rdiff)
ggrid_id_patchpool_np[800,800]
grid_ids_da.visualize(engine="cytoscape")
ggrid_id, ggrid_patchpool = creategrid(fday, exfeat)
ggrid_id, ggrid_patchpool = creategrid_np(fday, exfeat)
ggrid_id, ggrid_patchpool = creategrid(fday, exfeat)
stats = featurestat(ggrid_patchpool)
stats1=stats[(~stats['feature'].str.contains('corine')) & (~stats['feature'].str.contains('dir_max')) & (~stats['feature'].str.contains('dom_dir'))]
stats1[(~stats1['feature'].str.contains('wkd')) & (~stats1['feature'].str.contains('month'))]
plotfeatures(4, 4, 1, ggrid_patchpool, ['corine', 'dom_dir', 'dir_max', 'wkd', 'month', 'res_max', 'fire', 'aspect', 'curvature'])
plotfeatture("evi", featdict, ggrid_patchpool)#, cmap='Spectral_r')
plotfeatures(4, 4, 1, ggrid_patchpool)
plotfeatures(4, 4, 17, ggrid_patchpool)
plotfeatures(4, 4, 33, ggrid_patchpool)
plotfeatures(4, 4, 49, ggrid_patchpool)
plotfeatures(4, 4, 65, ggrid_patchpool)
plotfeatures(4, 4, 81, ggrid_patchpool)
'''