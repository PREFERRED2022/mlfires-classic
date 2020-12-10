import numpy as np
import pandas as pd
import gc
import os
from pathlib import Path
from ast import literal_eval
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import precision_score, make_scorer,recall_score,f1_score,roc_auc_score

def hyperparameter_tune(base_model, parameters, kfold, X, Y,groups):
    k = StratifiedKFold(n_splits=kfold, shuffle=False)

    prec_1 = make_scorer(precision_score, pos_label=1)
    rec_1 = make_scorer(recall_score, pos_label=1)
    f1_1 = make_scorer(f1_score, pos_label=1)
    roc = make_scorer(roc_auc_score)

    prec_0 = make_scorer(precision_score, pos_label=0)
    rec_0 = make_scorer(recall_score, pos_label=0)
    f1_0 = make_scorer(f1_score, pos_label=0)

    metrics = {'prec_1': prec_1, 'rec_1': rec_1, 'f1_1': f1_1, 'roc': roc, 'prec_0': prec_0, 'rec_0': rec_0,
               'f1_0': f1_0}

    optimal_model = RandomizedSearchCV(base_model, parameters,scoring=metrics, n_iter=1, cv=k, verbose=1,refit='rec_1', return_train_score=True)
    optimal_model.fit(X, Y,groups)

    return optimal_model.best_params_, optimal_model.best_score_, optimal_model.cv_results_


#data = pd.read_csv('/home/sgirtsou/Documents/ML-dataset_newLU/dataset_dummies.csv')
data = pd.read_csv('/home/sgirtsou/Documents/ML-dataset_newLU/dataset_corine_level2_onehotenc.csv')

X = data[['max_temp', 'min_temp', 'mean_temp', 'res_max', 'dom_vel', 'rain_7days', 'dem', 'slope',
       'curvature', 'aspect','ndvi_new', 'evi','dir_max_1','dir_max_2', 'dir_max_3', 'dir_max_4', 'dir_max_5', 'dir_max_6',
       'dir_max_7', 'dir_max_8', 'dom_dir_1', 'dom_dir_2', 'dom_dir_3','dom_dir_4', 'dom_dir_5', 'dom_dir_6', 'dom_dir_7', 'dom_dir_8',
       'corine_2_11', 'corine_2_12', 'corine_2_13', 'corine_2_14','corine_2_21', 'corine_2_22', 'corine_2_23', 'corine_2_24',
       'corine_2_31', 'corine_2_32', 'corine_2_33', 'corine_2_41','corine_2_42', 'corine_2_51', 'corine_2_52']]
Y = data['fire']

model = ExtraTreesClassifier(n_jobs =4)
groups = data['firedate']
groupskfold = groups.values

parameters = {
    'n_estimators' :[10, 20, 40, 60, 80, 100, 200, 400, 600, 800, 1000],
    'criterion' : ['gini', 'entropy'],
    'max_depth' : range(2, 40, 2),
    'min_samples_split': [2, 10, 50, 70,100,120,150,180, 200, 250,400,600,1000, 1300, 2000],
    'min_samples_leaf': [5, 10, 15, 20, 25, 30, 35, 40, 45],
    'max_features': list(range(1,X.shape[1])),
    'bootstrap': [True, False],
    'oob_score': [True, False],
    'class_weight': [{0:4,1:6},{0:1,1:10},{0:1,1:50},{0:1,1:70}]
}

best_scores = []
best_parameters = []
full_scores = []

folds =[10]


columns_sel = ['param_n_estimators', 'param_max_features', 'param_max_depth',
               'param_criterion','param_bootstrap', 'params', 'mean_test_acc', 'mean_train_acc', 'mean_test_AUC', 'mean_train_AUC',
               'mean_test_prec', 'mean_train_prec', 'mean_test_rec', 'mean_train_rec', 'rank_test_f_score', 'mean_train_f_score','folds']

results = pd.DataFrame(columns=columns_sel)

for i in folds:
    print("\ncv = ", i)
    best_params, best_score, full_scores = hyperparameter_tune(model, parameters, i, X, Y,groupskfold)

    df_results = pd.DataFrame.from_dict(full_scores)
    df_results['folds'] = int(i)
    df_sel = df_results[['mean_train_precision','std_train_precision','mean_test_precision','std_test_precision','mean_train_recall','std_train_recall','mean_test_recall',
                             'std_test_recall','mean_train_f1', 'std_train_f1', 'mean_test_f1', 'std_test_f1', 'params','folds']]
    df_results.to_csv('/home/sgirtsou/Documents/GridSearchCV/ExtraTrees/split'+str(i)+'_corine_l2_groups.csv')
    df_sel.to_csv('/home/sgirtsou/Documents/GridSearchCV/ExtraTrees/split_sh' + str(i) + '_corine_l2_groups.csv')
    '''
    df1 = df_results[columns_sel]
    df_no_split_cols = [c for c in df_results.columns if 'split' not in c]

    df_results.to_csv('rfresults.csv')
    df_results[df_no_split_cols].to_csv('rfresults_nosplit.csv')

    results = pd.concat([results, df1])

    best_scores.append(best_score)
    best_parameters.append(best_params)
    '''