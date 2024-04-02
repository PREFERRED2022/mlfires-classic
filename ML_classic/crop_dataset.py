import pandas as pd
import geopandas as gpd
import multiprocessing as mp
import time
import os
import fileutils

'''
periphereies
0 	Π. ΑΝΑΤΟΛΙΚΗΣ ΜΑΚΕΔΟΝΙΑΣ - ΘΡΑΚΗΣ 	
1 	Π. ΚΕΝΤΡΙΚΗΣ ΜΑΚΕΔΟΝΙΑΣ 	
2 	Π. ΔΥΤΙΚΗΣ ΜΑΚΕΔΟΝΙΑΣ 	
3 	Π. ΗΠΕΙΡΟΥ 	MULTIPOLYGON 
4 	Π. ΘΕΣΣΑΛΙΑΣ 	
5 	Π. ΒΟΡΕΙΟΥ ΑΙΓΑΙΟΥ 	
6 	Π. ΝΟΤΙΟΥ ΑΙΓΑΙΟΥ 	
7 	Π. ΣΤΕΡΕΑΣ ΕΛΛΑΔΑΣ 	
8 	Π. ΔΥΤΙΚΗΣ ΕΛΛΑΔΑΣ 	
9 	Π. ΠΕΛΟΠΟΝΝΗΣΟΥ 	
10 	Π. ΙΟΝΙΩΝ ΝΗΣΩΝ 	
11 	Π. ΚΡΗΤΗΣ 	
12 	Π. ΑΤΤΙΚΗΣ 	
'''
'''
walk folders to find all dataset files
'''
def walkmonthdays(sfolder, pattern, walktype='walk'):
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

def cropfile(f, cropfolder, gdfperif, suffix="", xcoord='x', ycoord='y', usexyid=False):
    fcrop = os.path.join(cropfolder, os.path.basename(f).split('.')[0] + suffix + '.csv')
    #if os.path.isfile(fcrop):
    #    return
    print('Start processing file %s'%f)
    df=pd.read_csv(f)
    if 'crs' in df.columns: df.drop(columns=['crs'], inplace=True)

    if usexyid:
        df[usexyid]=df[usexyid].astype(str)
        df['xtemp'] = df[usexyid].str.slice(0, 6).astype(int) / 10000
        df['ytemp'] = df[usexyid].str.slice(6, 12).astype(int) / 10000
    else:
        df['xtemp'] = df[xcoord]
        df['ytemp'] = df[ycoord]

    geom = gpd.points_from_xy(df['xtemp'], df['ytemp'])
    gdf = gpd.GeoDataFrame(df, geometry=geom)
    gdf = gdf.set_crs(4326)
    gdf_crop = gpd.sjoin(gdf, gdfperif, how='inner', predicate='within').drop(columns=['xtemp','ytemp'])
    df_crop = pd.DataFrame(gdf_crop.drop(columns=['geometry','PER','index_right']))
    df_crop.to_csv(fcrop,index=False)
    print('Done processing file. Output %s' % fcrop)

def getperif(perif=None):
    gdfperif = gpd.read_file(r'/mnt/nvme2tb/ffp/datasets/test/2019/perif/periphereies.shp', encoding='Windows-1253')
    gdfperif = gdfperif.to_crs(4326)
    return gdfperif

