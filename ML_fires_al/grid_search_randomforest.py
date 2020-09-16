import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import randint
import matplotlib.pyplot as plt
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


def hyperparameter_tune(base_model, parameters, n_iter, kfold, X, y):
    start_time = time.time()

    # Arrange data into folds with approx equal proportion of classes within each fold
    k = StratifiedKFold(n_splits=kfold, shuffle=False)

    optimal_model = GridSearchCV(base_model,
                                       param_grid = parameters,
                                       #n_iter=n_iter,
                                       cv=k,
                                       n_jobs=-1)
                                       #random_state=SEED)

    optimal_model.fit(X, y)

    stop_time = time.time()

    scores = cross_val_score(optimal_model, X, y, cv=k, scoring="accuracy")

    print("Elapsed Time:", time.strftime("%H:%M:%S", time.gmtime(stop_time - start_time)))
    print("====================")
    print("Cross Val Mean: {:.3f}, Cross Val Stdev: {:.3f}".format(scores.mean(), scores.std()))
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
depth = list(range(50, 501, 25))
depth.append(None)
lots_of_parameters = {
    #"max_depth": depth, #depth of each tree
    "n_estimators": list(range(50, 501, 25)), #trees of the forest
    #"n_estimators": list(range(50, 101, 50)),
    #"max_features": list(range(1,X_.shape[1])),
    "criterion": ["gini", "entropy"],
    #"oob_score": [True, False],
    "bootstrap": [True, False]
    #"min_samples_leaf": 1 #randint(1, 4) #minimum number of samples required to be at a leaf node
}

#best_params, best_score = hyperparameter_tune(rf, lots_of_parameters, 10, 5, X_, y_)

best_scores = []
best_params = []
full_scores = []
folds = range(2, 8)

for i in folds:
    print("\ncv = ", i)
    best_params, best_score, full_scores = hyperparameter_tune(rf, lots_of_parameters, 10, i,  X_, y_)
    best_scores.append(best_score)
    best_params.append(best_params)
    full_scores.append(full_scores)

with open('/home/sgirtsou/Documents/GridSearchCV/CV_rf.csv', 'rw', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(full_scores)

with open('/home/sgirtsou/Documents/GridSearchCV/CV_rf_bestparams.csv', 'rw', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(best_params)

with open('/home/sgirtsou/Documents/GridSearchCV/CV_rf_bestscores.csv', 'rw', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(best_params)

X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=0.10, stratify=y)