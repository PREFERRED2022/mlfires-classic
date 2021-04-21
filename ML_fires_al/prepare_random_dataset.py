import pandas as pd
import normdataset
import check_and_prepare_dataset
from datetime import datetime
import numpy as np

def create_time_columns(sdate, out = "wkd"):
    dt = datetime.strptime(sdate, '%Y-%m-%d')
    if out == "wkd":
        return dt.weekday()
    elif out ==  "month":
        return dt.month

#print(create_time_columns("2021-04-20"))

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

Xnorm = normdataset.normalize_dataset(X, aggrfile = '/home/sgirtsou/PycharmProjects/ML/ML_fires_al/stats/featurestats.json')
pd.concat([dates, Xnorm, y], axis=1).to_csv('/home/aapos/Documents/newcrossval/datasets/randomnofire/old_random_new_features_norm.csv')
