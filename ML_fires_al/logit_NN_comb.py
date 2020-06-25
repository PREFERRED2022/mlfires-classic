import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np


def lb_nn_comb(lb, nn):
    proba = max(lb, nn) if lb > 0.98 else nn
    class1 = 1 if proba >= 0.5 else 0
    return proba, class1


os.chdir('/home/sgirtsou/Documents/June2019/LB_results')
mypath = '/home/sgirtsou/Documents/June2019/LB_results'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

cn = 0


def dayevents(dayfile, df_day, fireids):
    fdate = dayfile[10:12] + "/" + dayfile[8:10] + "/" + dayfile[4:8]
    fireids = fireids[fireids['firedate_g'] == fdate]
    df_day['id'] = df_day['id'].astype(int)
    df_day = df_day.join(fireids.set_index('id'), on='id', how='left')
    return df_day


ids = '/home/sgirtsou/Documents/ML-dataset_newLU/fires_2019_firehubcellsid.csv'
fireids = pd.read_csv(ids)
for fname in onlyfiles:
    if fname.endswith('_lb.csv') and 'comb' not in fname:
        print(fname)
        df_greece = pd.read_csv(fname)
        df_greece = dayevents(fname, df_greece, fireids)
        # cn+=len(df_greece[(df_greece.Class_pred_lb == 1) & (df_greece.Class_pred == 0) & (df_greece.fire == 1)])
        # Class_pred, Class_0_proba,Class_1_proba,Class_pred_lb, Class_0_proba_lb, Class_1_proba_lb
        if len(df_greece) != 373544:
            print("wrong num of rows in %s" % fname)
        probas = df_greece[['Class_1_proba_lb', 'Class_1_proba', 'fire']].to_numpy()
        # [1:]
        '''
        lb_probas = df_greece[['Class_1_proba_lb']].to_numpy()[1:]
        nn_probas = df_greece[['Class_1_proba']].to_numpy()[1:]
        '''
        v_lb_nn_comb = np.vectorize(lb_nn_comb)
        comb_probas = v_lb_nn_comb(probas[:, 0], probas[:, 1])
        # cn+=len(probas[(probas[:,0] >=0.98) & (probas[:,1] < 0.5) & (probas[:,2] == 1)])
        allprobas = np.vstack([comb_probas[0], probas[:, 2]])
        allprobas = np.transpose(allprobas)
        #cn += len(allprobas[(allprobas[:, 0] >= 0.5) & (allprobas[:, 1] == 1)])
        #print(cn)
        if np.any(np.isnan(comb_probas[0])) or np.any(np.isnan(comb_probas[1])):
            print("comb has nan in %s" % fname)

        # comb_probas_df = pd.DataFrame({'Comb_proba_1': comb_probas[0], 'Comb_class_pred': comb_probas[1]})
        df_greece['Comb_proba_1'] = pd.Series(comb_probas[0])
        df_greece['Comb_class_pred'] = pd.Series(comb_probas[1])
        # df_results = pd.concat([df_greece, comb_probas_df], axis=1)
        basedir = '/home/sgirtsou/Documents/June2019/'
        comb_results = os.path.join(basedir, 'Comb_results')
        if not os.path.exists(comb_results):
            os.makedirs(comb_results)
        newcsv=os.path.join(comb_results, fname.split('_lb')[0] + '_comb.csv')
        if not os.path.isfile(newcsv):
            df_greece.to_csv(newcsv)
