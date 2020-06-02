import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np

def lb_nn_comb(lb, nn):
    proba=max(lb, nn) if lb > 0.98 else nn
    class1= 1 if proba>=0.5 else 0
    return proba, class1

os.chdir('/home/sgirtsou/Documents/ML-dataset_newLU/csvs_withfire_results/LB_results')
mypath = '/home/sgirtsou/Documents/ML-dataset_newLU/csvs_withfire_results/LB_results'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

cn = 0
for fname in onlyfiles:
    if fname.endswith('csv') and 'comb' not in fname:
        df_greece = pd.read_csv(fname)
        #cn+=len(df_greece[(df_greece.Class_pred_lb == 1) & (df_greece.Class_pred == 0) & (df_greece.fire == 1)])
        # Class_pred, Class_0_proba,Class_1_proba,Class_pred_lb, Class_0_proba_lb, Class_1_proba_lb
        if len(df_greece)!=373544:
            print("wrong num of rows in %s" % fname)
        probas = df_greece[['Class_1_proba_lb','Class_1_proba','fire']].to_numpy()
        #[1:]
        '''
        lb_probas = df_greece[['Class_1_proba_lb']].to_numpy()[1:]
        nn_probas = df_greece[['Class_1_proba']].to_numpy()[1:]
        '''
        v_lb_nn_comb=np.vectorize(lb_nn_comb)
        comb_probas=v_lb_nn_comb(probas[:,0],probas[:,1])
        #cn+=len(probas[(probas[:,0] >=0.98) & (probas[:,1] < 0.5) & (probas[:,2] == 1)])
        allprobas=np.vstack([comb_probas[0], probas[:,2]])
        allprobas=np.transpose(allprobas)
        cn+=len(allprobas[(allprobas[:, 0] >=0.5) & (allprobas[:, 1] == 1)])
        print(cn)
        if np.any(np.isnan(comb_probas[0])) or np.any(np.isnan(comb_probas[1])):
            print("comb has nan in %s" % fname)

        #comb_probas_df = pd.DataFrame({'Comb_proba_1': comb_probas[0], 'Comb_class_pred': comb_probas[1]})
        df_greece['Comb_proba_1'] = pd.Series(comb_probas[0])
        df_greece['Comb_class_pred'] = pd.Series(comb_probas[1])
        #df_results = pd.concat([df_greece, comb_probas_df], axis=1)
        df_greece.to_csv('/home/sgirtsou/Documents/ML-dataset_newLU/csvs_withfire_results/LB_results/' + fname[0:12] + '_comb.csv')
