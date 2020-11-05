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
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_fscore_support

def hyperparameter_tune(base_model, parameters, kfold, X, Y):
    k = StratifiedKFold(n_splits=kfold, shuffle=False)
    metrics = ['precision', 'recall', 'f1']

    optimal_model = RandomizedSearchCV(base_model, parameters,scoring=metrics, n_iter=200, cv=k, verbose=1,refit='recall', return_train_score=True)
    optimal_model.fit(X, Y)

    return optimal_model.best_params_, optimal_model.best_score_, optimal_model.cv_results_


data = pd.read_csv('/home/sgirtsou/Documents/ML-dataset_newLU/dataset_dummies.csv')

X = data[['max_temp', 'min_temp', 'mean_temp',
       'res_max', 'dom_vel', 'rain_7days', 'Near_dist', 'DEM', 'Slope',
       'Curvature', 'Aspect', 'ndvi_new', 'evi', 'dir_max_1',
       'dir_max_2', 'dir_max_3', 'dir_max_4', 'dir_max_5', 'dir_max_6',
       'dir_max_7', 'dir_max_8', 'dom_dir_1', 'dom_dir_2', 'dom_dir_3',
       'dom_dir_4', 'dom_dir_5', 'dom_dir_6', 'dom_dir_7', 'dom_dir_8',
       'fclass_bridleway', 'fclass_footway', 'fclass_living_street',
       'fclass_motorway', 'fclass_path', 'fclass_pedestrian', 'fclass_primary',
       'fclass_residential', 'fclass_secondary', 'fclass_service',
       'fclass_steps', 'fclass_tertiary', 'fclass_track',
       'fclass_track_grade1', 'fclass_track_grade2', 'fclass_track_grade3',
       'fclass_track_grade4', 'fclass_track_grade5', 'fclass_trunk',
       'fclass_unclassified', 'fclass_unknown', 'Corine_111', 'Corine_112',
       'Corine_121', 'Corine_122', 'Corine_123', 'Corine_131', 'Corine_132',
       'Corine_133', 'Corine_142', 'Corine_211', 'Corine_212', 'Corine_213',
       'Corine_221', 'Corine_222', 'Corine_223', 'Corine_231', 'Corine_241',
       'Corine_242', 'Corine_243', 'Corine_311', 'Corine_312', 'Corine_313',
       'Corine_321', 'Corine_322', 'Corine_323', 'Corine_324', 'Corine_331',
       'Corine_332', 'Corine_333', 'Corine_334', 'Corine_411', 'Corine_421',
       'Corine_511', 'Corine_512', 'Corine_523']]
Y = data['fire']

model = ExtraTreesClassifier(n_jobs =4)


parameters = {
    'n_estimators' :[10, 20, 40, 60, 80, 100, 200, 400, 600, 800, 1000],
    'criterion' : ['gini', 'entropy'],
    'max_depth' : range(2, 40, 2),
    'min_samples_split': [2, 10, 50, 70,100,120,150,180, 200, 250,400,600,1000, 1300, 2000],
    'min_samples_leaf': [5, 10, 15, 20, 25, 30, 35, 40, 45],
    'max_features': list(range(1,X.shape[1])),
    'bootstrap': [True, False],
    'oob_score': [True, False],
    'class_weight': [{0:4,1:6},{0:3,1:7},{0:2,1:8},{0:1,1:9}]
}

best_scores = []
best_parameters = []
full_scores = []

folds =[9, 10]


columns_sel = ['param_n_estimators', 'param_max_features', 'param_max_depth',
               'param_criterion','param_bootstrap', 'params', 'mean_test_acc', 'mean_train_acc', 'mean_test_AUC', 'mean_train_AUC',
               'mean_test_prec', 'mean_train_prec', 'mean_test_rec', 'mean_train_rec', 'rank_test_f_score', 'mean_train_f_score','folds']

results = pd.DataFrame(columns=columns_sel)

for i in folds:
    print("\ncv = ", i)
    best_params, best_score, full_scores = hyperparameter_tune(model, parameters, i, X, Y)

    df_results = pd.DataFrame.from_dict(full_scores)
    df_results['folds'] = int(i)
    df_sel = df_results[['mean_train_precision','std_train_precision','mean_test_precision','std_test_precision','mean_train_recall','std_train_recall','mean_test_recall',
                             'std_test_recall','mean_train_f1', 'std_train_f1', 'mean_test_f1', 'std_test_f1', 'params','folds']]
    df_results.to_csv('/home/sgirtsou/Documents/GridSearchCV/ExtraTrees/split'+str(i)+'.csv')
    df_sel.to_csv('/home/sgirtsou/Documents/GridSearchCV/ExtraTrees/split_sh' + str(i) + '.csv')
    '''
    df1 = df_results[columns_sel]
    df_no_split_cols = [c for c in df_results.columns if 'split' not in c]

    df_results.to_csv('rfresults.csv')
    df_results[df_no_split_cols].to_csv('rfresults_nosplit.csv')

    results = pd.concat([results, df1])

    best_scores.append(best_score)
    best_parameters.append(best_params)
    '''