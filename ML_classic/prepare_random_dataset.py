import pandas as pd
import normdataset
import check_and_prepare_dataset
from datetime import datetime
import numpy as np
import re
import csv
import bisect

def create_time_columns(sdate, out = "wkd"):
    dt = datetime.strptime(sdate, '%Y-%m-%d')
    if out == "wkd":
        return dt.weekday()
    elif out == "month":
        return dt.month

def index(a, x):
    'Locate the leftmost value exactly equal to x'
    i = bisect.bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    return -1

def readxy():
    xylist = []
    idlist = []
    with open('/home/aapos/Documents/newcrossval/april_2010_norm.csv', 'r') as fin:
        reader = csv.reader(fin)
        headers = next(reader)
        i=0
        for row in reader:
            id = int(float(row[2]))
            if index(idlist, id) == -1:
                bisect.insort(idlist, id)
                xylist += [{'id': id, 'x':row[-2], 'y':row[-1]}]
            if i % 1000000 == 0:
                print('row: %d'%i)
            i+=1
            #if i>10000:
            #    break
    return xylist

def correctcatcolumns(X, basecol, allclasses):
    cols = [c for c in X.columns if basecol in c]
    existclasses = {int(float(re.search("(?<=%s).*$"%basecol, c).group(0))):c for c in cols}
    for classnum in allclasses:
        newcol = 'bin_%s%d'%(basecol,classnum)
        if classnum not in existclasses.keys():
            X[newcol] = 0
        else:
            X.rename(columns={existclasses[classnum]: newcol}, inplace=True)
    for existclass in existclasses:
        if existclass not in allclasses:
            X = X.drop(columns=[existclasses[existclass]])
    return X

#print(create_time_columns("2021-04-20"))

# collect x y features
#xy = readxy()
#dfxy = pd.DataFrame(xy)
#dfxy.to_csv('/home/aapos/Documents/newcrossval/xyid_norm.csv', index=False)

# load id
iddf = pd.read_csv('/home/aapos/Documents/newcrossval/datasets/randomnofire/training_dataset_dew_lst_dummies.csv', usecols=['id'], dtype={'id': np.int32})

# add frequency features
df_freq = pd.read_csv('/home/aapos/Documents/newcrossval/datasets/randomnofire/frequencyfeature_density_40km_1970_2009.csv', dtype={'id': np.int32})
df_initial = pd.read_csv('/home/aapos/Documents/newcrossval/datasets/randomnofire/training_dataset_dew_lst_dummies.csv', dtype={'id': np.int32})
df = pd.merge(df_initial, df_freq, on='id', how='inner')
df = df.rename(columns={'81':'f81'})
df.to_csv('/home/aapos/Documents/newcrossval/datasets/randomnofire/old_random_new_features.csv', index=False)
X, y, dates = check_and_prepare_dataset.load_dataset('/home/aapos/Documents/newcrossval/datasets/randomnofire/old_random_new_features.csv')

# add time features
tfeat = [{'col':'wkd', 'range':range(0,7)}, {'col':'month','range':(1,13)}]
Xtime = pd.DataFrame()
for feat in tfeat:
    Xtime[feat['col']] = dates.apply(lambda d: create_time_columns(d, feat['col']))
Xtime = pd.concat([dates, Xtime], axis=1)

for feat in tfeat:
    Xbin = pd.get_dummies(Xtime[feat['col']])
    for c in Xbin.columns:
        Xbin = Xbin.rename(columns={c: 'bin_'+feat['col']+'_'+'%s'%c})
    X = pd.concat([X, Xbin], axis = 1)

#correct corine
allcorineclass = [111, 112, 121, 122, 123, 124, 131, 132, 133, 141, 142, 211, 212, 213, 221, 222, 223, 231, 241, 242,
243, 244, 311, 312, 313, 321, 322, 323, 324, 331, 332, 333, 334, 411, 412, 421, 422, 511, 512, 521]
X = correctcatcolumns(X, 'corine_', allcorineclass)
#correct dirmax, domdir
alldirclass = [1,2,3,4,5,6,7,8]
X = correctcatcolumns(X, 'dir_max_', alldirclass)
X = correctcatcolumns(X, 'dom_dir_', alldirclass)

Xnorm = normdataset.normalize_dataset(X, aggrfile = '/home/sgirtsou/PycharmProjects/ML/ML_fires_al/stats/featurestats.json')

dfxy = pd.read_csv('/home/aapos/Documents/newcrossval/xyid_norm.csv')
Xnorm = pd.concat([iddf, dates, Xnorm, y], axis=1)
Xnorm = pd.merge(Xnorm, dfxy, on='id', how='inner')
Xnorm.to_csv('/home/aapos/Documents/newcrossval/datasets/randomnofire/old_random_new_features_norm.csv', index=False)
