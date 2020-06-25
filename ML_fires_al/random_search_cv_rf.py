import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
import time
import warnings
import csv

df = pd.read_csv('/home/sgirtsou/Documents/ML-dataset_newLU/training_dataset.csv')

def normalized_values(y, dfmax, dfmin, dfmean, dfstd, t=None):
    if not t:
        a = (y - dfmin) / (dfmax - dfmin)
        return (a)
    elif t == 'std':
        a = (y - dfmean) / dfstd
        return (a)
    elif t == 'no':
        return y


def normalize_dataset(X_unnorm_int, norm_type=None):
    X = pd.DataFrame()
    for c in X_unnorm_int.columns:
        print(c)
        dfmax = X_unnorm_int[c].max()
        dfmin = X_unnorm_int[c].min()
        dfmean = X_unnorm_int[c].mean()
        dfstd = X_unnorm_int[c].std()
        X[c] = X_unnorm_int.apply(lambda x: normalized_values(x[c], dfmax, dfmin, dfmean, dfstd, norm_type), axis=1)
    return X


def hyperparameter_tune(base_model, parameters, kfold, X, y):
    start_time = time.time()

    # Arrange data into folds with approx equal proportion of classes within each fold
    k = StratifiedKFold(n_splits=kfold, shuffle=False)

    scoring_st = {'acc': 'accuracy',
                  'AUC': 'roc_auc',
                  'prec': 'precision',
                  'rec' : 'recall',
                  'f_score' : 'f1'
    }
    optimal_model = RandomizedSearchCV(base_model,
                                       param_distributions=parameters,
                                       n_iter=200,
                                       cv=k,
                                       scoring = scoring_st,
                                       n_jobs=6,
                                       refit='rec',
                                       return_train_score=True)
                                       #random_state=SEED)

    optimal_model.fit(X, y)

    stop_time = time.time()



    #scores = cross_validate(optimal_model, X, y, cv=k, scoring= scoring_st, return_train_score=True, return_estimator=True)

    print("Elapsed Time:", time.strftime("%H:%M:%S", time.gmtime(stop_time - start_time)))
    print("====================")
    #print("Cross Val Mean: {:.3f}, Cross Val Stdev: {:.3f}".format(scores.mean(), scores.std()))
    print("Best Score: {:.3f}".format(optimal_model.best_score_))
    print("Best Parameters: {}".format(optimal_model.best_params_))


    return optimal_model.best_params_, optimal_model.best_score_, optimal_model.cv_results_

df_part = df[
    ['id', 'max_temp', 'min_temp', 'mean_temp', 'res_max', 'dir_max', 'dom_vel', 'dom_dir', 'rain_7days', 'Corine',
     'Slope', 'DEM', 'Curvature', 'Aspect', 'ndvi', 'fire']].copy()

X_unnorm, y_int = df_part[
                      ['max_temp', 'min_temp', 'mean_temp', 'res_max', 'dir_max', 'dom_vel', 'dom_dir', 'rain_7days',
                       'Corine', 'Slope', 'DEM', 'Curvature', 'Aspect', 'ndvi']], df_part['fire']

X = normalize_dataset(X_unnorm, 'std')
y = y_int

X_ = X.values
y_ = y.values
y.shape, X.shape, type(X_), type(y_)

rf = RandomForestClassifier(n_jobs=-1)
depth = [10, 20, 100, 200, 500, 1000, 2000, None] #list(range(25, 201, 25))
depth.append(None)
n_estimators = [50, 100, 150, 250, 350, 500, 750, 1000, 1500]#list(range(50, 501, 25))
min_samples_split = [2, 10, 50, 100, 200, 1000, 2000] #with numbers
#min_samples_leaf = [1, 10, 30] #with numbers
max_features = list(range(1,X_.shape[1]))

lots_of_parameters = {
    "max_depth": depth, #depth of each tree
    "n_estimators": n_estimators, #trees of the forest
   # "min_samples_leaf": min_samples_leaf,
    "criterion": ["gini", "entropy"],
    "max_features": max_features,
    "bootstrap": [True, False]
}

best_scores = []
best_parameters = []
full_scores = []
folds = range(2, 8)

columns_sel = ['param_n_estimators', 'param_max_features', 'param_max_depth',
               'param_criterion','param_bootstrap', 'params', 'mean_test_acc', 'mean_train_acc', 'mean_test_AUC', 'mean_train_AUC',
               'mean_test_prec', 'mean_train_prec', 'mean_test_rec', 'mean_train_rec', 'rank_test_f_score', 'mean_train_f_score','folds']

results = pd.DataFrame(columns=columns_sel)

for i in folds:
    print("\ncv = ", i)
    best_params, best_score, full_scores = hyperparameter_tune(rf, lots_of_parameters, i, X_, y_)

    df_results = pd.DataFrame.from_dict(full_scores)
    df_results['folds'] = int(i)
    df1 = df_results[columns_sel]

    df_results.to_csv('/home/sgirtsou/Documents/GridSearchCV/random_search2/rscv_' + str(i) + '.csv')

    results = pd.concat([results, df1])

    best_scores.append(best_score)
    best_parameters.append(best_params)

results.to_csv('/home/sgirtsou/Documents/GridSearchCV/random_search2/rscv_' + str(i) + '.csv')
i = 1