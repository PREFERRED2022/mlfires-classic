#!/usr/bin/env python
from hyperopt import Trials, fmin, tpe, hp, STATUS_OK
from pandas import read_csv
from sklearn.model_selection import train_test_split, KFold, GroupKFold
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import tensorflow.keras.metrics
from tensorflow.keras.utils import to_categorical
import numpy as np
from pandas import concat
from pandas import DataFrame
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd
import normdataset
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow import summary
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
import time
import json
import space

num_folds = 10
# kf = KFold(n_splits=num_folds, shuffle=True)
kf = GroupKFold(n_splits=num_folds)
random_state = 42

def drop_all0_features(df):
    for c in df.columns:
        if 'bin' in c:
            u = df[c].unique()
            if (len(u)>1):
                cnt = df.groupby([c]).count()
                if cnt["max_temp"][1]<20:
                    print("Droping Feature %s, count: %d"%(c, cnt["max_temp"][1]))
                    df.drop(columns=[c])
            else:
                print("%s not exists (size : %d)"%(c, len(u)))

def prepare_dataset(df, X_columns, y_columns, firedate_col, corine_col, domdir_col, dirmax_col):
    # df = read_csv('/home/sgirtsou/Documents/ML-dataset_newLU/training_dataset.csv')
    df = df.dropna()
    #df.columns = ['id', 'firedate_x', 'max_temp', 'min_temp', 'mean_temp', 'res_max',
    #              'dir_max', 'dom_vel', 'dom_dir', 'rain_7days', 'Corine', 'Forest',
    #              'fire', 'firedate_g', 'firedate_y', 'tile', 'max_temp_y', 'DEM',
    #              'Slope', 'Curvature', 'Aspect', 'image', 'ndvi']
    #df_part = df[
    #    ['id', 'max_temp', 'min_temp', 'mean_temp', 'res_max', 'dir_max', 'dom_vel', 'dom_dir', 'rain_7days', 'Corine',
    #     'Slope', 'DEM', 'Curvature', 'Aspect', 'ndvi', 'fire']]

    X_unnorm, y_int = df[X_columns], df[y_columns]

    # categories to binary
    Xbindomdir = pd.get_dummies(X_unnorm[domdir_col].round())
    if 0 in Xbindomdir.columns:
        del Xbindomdir[0]
    ddircols = []
    for i in range(1, 9):
        ddircols.append('binDDIR_%d' % i)
    Xbindomdir.columns = ddircols
    Xbindirmax = pd.get_dummies(X_unnorm[dirmax_col].round())
    if 0 in Xbindirmax.columns:
        del Xbindirmax[0]
    dmaxcols = []
    for i in range(1, 9):
        dmaxcols.append('binMDIR_%d' % i)
    Xbindirmax.columns = dmaxcols

    Xbincorine = pd.get_dummies(X_unnorm[corine_col])
    corcols = ['binCOR' + str(c) for c in Xbincorine.columns]
    Xbincorine.columns = corcols

    X_unnorm = pd.concat([X_unnorm, Xbindomdir, Xbindirmax, Xbincorine], axis=1)
    del X_unnorm[corine_col]
    del X_unnorm[domdir_col]
    del X_unnorm[dirmax_col]

    #str_classes = ['Corine']
    #X_unnorm_int = normdataset.index_string_values(X_unnorm, str_classes)
    #X = normdataset.normalize_dataset(X_unnorm_int, 'std')

    X = normdataset.normalize_dataset(X_unnorm)
    y = y_int
    groupspd = df[firedate_col]

    return X, y, groupspd

# load the dataset
def load_dataset():
    dsetfolder = 'data/'
    #dsfile = 'dataset_ndvi_lu.csv'
    dsfile = 'dataset_1_10.csv'
    X_columns = ['max_temp', 'min_temp', 'mean_temp', 'res_max', 'dir_max', 'dom_vel', 'dom_dir',
                 'rain_7days',
                 'Corine', 'Slope', 'DEM', 'Curvature', 'Aspect', 'ndvi']
    y_columns = ['fire']
    dsreadyfile = 'dataset_nn_ready.csv'
    if not os.path.exists(os.path.join(dsetfolder,dsreadyfile)):
        df = pd.read_csv(os.path.join(dsetfolder, dsfile))
        X_columns_upper = [c.upper() for c in X_columns]
        newcols = [c for c in df.columns if c.upper() in X_columns_upper or any([cX in c.upper() for cX in X_columns_upper])]
        X_columns = newcols
        corine_col = [c for c in df.columns if 'Corine'.upper() in c.upper()][0]
        dirmax_col = [c for c in df.columns if 'dir_max'.upper() in c.upper()][0]
        domdir_col = [c for c in df.columns if 'dom_dir'.upper() in c.upper()][0]
        firedate_col = [c for c in df.columns if 'firedate'.upper() in c.upper()][0]
        X, y, groupspd = prepare_dataset(df, X_columns, y_columns, firedate_col, corine_col, domdir_col, dirmax_col)
        featdf = pd.concat([X, y, groupspd], axis=1)
        featdf[[c for c in featdf.columns if 'Unnamed' not in c]].to_csv(os.path.join(dsetfolder, dsreadyfile))
    else:
        featdf = pd.read_csv(os.path.join(dsetfolder, dsreadyfile))
        firedate_col = [c for c in featdf.columns if 'firedate'.upper() in c.upper()][0]
        X_columns_new = [c for c in featdf.columns if c not in [firedate_col,'fire'] and 'Unnamed' not in c]
        X = featdf[X_columns_new]
        y = featdf[y_columns]
        groupspd = featdf[firedate_col]


    #drop_all0_features(featdf)

    return X, y, groupspd


X_pd, y_pd, groups_pd = load_dataset()


def create_NN_model(params, X):
    # define model
    model = Sequential()
    n_features = X.shape[1]
    intlayers = int(params['n_internal_layers'][0])
    model.add(Dense(params['n_internal_layers'][1]['layer_1_' + str(intlayers) + '_nodes'], activation='relu',
                    input_shape=(n_features,)))
    for i in range(2, intlayers + 2):
        model.add(Dense(int(params['n_internal_layers'][1]['layer_' + str(i) + '_' + str(intlayers) + '_nodes']),
                        activation='relu'))
        # model.add(Dense(1, activation='sigmoid'))
    model.add(Dense(2, activation='softmax'))

    # compile the model

    from tensorflow.keras.optimizers import Adam

    adam = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)

    if params['metric'] == 'accuracy':
        metrics = ['accuracy']
    elif params['metric'] == 'sparse':
        metrics = [tensorflow.metrics.SparseCategoricalAccuracy()]
    #elif params['metric'] == 'tn':
        #metrics = [tensorflow.metrics.TrueNegatives(),tensorflow.metrics.TruePositives()]
    model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=metrics)  # , AUC(multi_label=False)])

    return model


def nnfit(params, cv=kf, X_pd=X_pd, y_pd=y_pd, groups_pd=groups_pd):
    # the function gets a set of variable parameters in "param"
    '''
    params = {'n_internal_layers': params['n_internal_layers'][0],
              'layer_1_nodes': params['n_internal_layers'][1],
              'layer_2_nodes': params['layer_2_nodes'],
              'layer_3_nodes': params['layer_3_nodes'],
              'layer_4_nodes': params['layer_4_nodes']}
              '''

    metrics = []
    cnt = 0
    print("NN params : %s" % params)

    if params['feature_drop']:
        X_pd=X_pd.drop(columns=[c for c in X_pd.columns if params['feature_drop'] in c])

    X = X_pd.values
    y = y_pd.values
    groups = groups_pd.values

    for train_index, test_index in cv.split(X, y, groups):
        cnt += 1
        print("Fitting Fold %d" % cnt)
        start_time = time.time()
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_val = X[train_index], X[test_index]
        y_train, y_val = y[train_index], y[test_index]
        y_val = y_val[:,0]
        y_train = y_train[:,0]
        model = create_NN_model(params, X)
        es = EarlyStopping(monitor='loss', patience=10, min_delta=0.002)
        start_time = time.time()
        res = model.fit(X_train, y_train, batch_size=512, epochs=max_epochs, verbose=0, validation_data=(X_val, y_val),\
                        callbacks=[es], class_weight=params['class_weights'])
        print("Fit time (min): %s"%((time.time() - start_time)/60.0))
        start_time = time.time()
        es_epochs = len(res.history['loss'])
        #print("epochs run: %d" % es_epochs)

        aucmetric = tensorflow.metrics.AUC()

        '''validation set metrics'''
        loss_test, acc_test = model.evaluate(X_val, y_val, batch_size=512, verbose=0)
        y_pred = model.predict_classes(X_val)
        y_scores = model.predict(X_val)

        aucmetric.update_state(y_val, y_scores[:,1])
        auc_val = float(aucmetric.result())

        prec_test = precision_score(y_val, y_pred)
        rec_test = recall_score(y_val, y_pred)
        f1_test = f1_score(y_val, y_pred)
        print("Validation metrics time (min): %s"%((time.time() - start_time)/60.0))
        start_time = time.time()

        '''training set metrics'''
        loss_train, acc_train = model.evaluate(X_train, y_train, batch_size=512, verbose=0)
        y_pred = model.predict_classes(X_train)
        y_scores = model.predict(X_train)
        aucmetric.update_state(y_train, y_scores[:,1])
        auc_train = float(aucmetric.result())

        prec_train = precision_score(y_train, y_pred)
        rec_train = recall_score(y_train, y_pred)
        f1_train = f1_score(y_train, y_pred)
        print("Training metrics time (min): %s"%((time.time() - start_time)/60.0))

        metrics.append(
            {'loss val.': loss_test, 'loss train': loss_train, 'accuracy val.': acc_test, 'accuracy train': acc_train,
             'precision val.': prec_test, 'precision train': prec_train, 'recall val.': rec_test,
             'recall train': rec_train,
             'f1-score val.': f1_test, 'f1-score train': f1_train, 'auc val.': auc_val,
             'auc train.': auc_train, 'early stop epochs': es_epochs})
        #print(metrics[-1])

    mean_metrics = {}
    for m in metrics[0]:
        mean_metrics[m] = sum(item.get(m, 0) for item in metrics) / len(metrics)
    print('Mean recall (on test) : %s' % mean_metrics['recall val.'])

    return {
        'loss': -mean_metrics['recall val.'],
        'status': STATUS_OK,
        # -- store other results like this
        # 'eval_time': time.time(),
        'metrics': mean_metrics,
        # -- attachments are handled differently
        # 'attachments':
        #    {'time_module': pickle.dumps(time.time)}
    }

space, max_trials, max_epochs = space.create_space()
trials = Trials()

best = fmin(fn=nnfit,  # function to optimize
            space=space,
            algo=tpe.suggest,  # optimization algorithm, hyperotp will select its parameters automatically
            max_evals=max_trials,  # maximum number of iterations
            trials=trials,  # logging
            rstate=np.random.RandomState(random_state)  # fixing random state for the reproducibility
            )

pd_opt = pd.DataFrame(columns=list(trials.trials[0]['result']['metrics'].keys()))
for t in trials:
    pdrow = t['result']['metrics']
    pdrow['params'] = str(t['misc']['vals'])
    pd_opt = pd_opt.append(pdrow, ignore_index=True)

hyp_res_base = 'hyperopt_results_'
cnt = 1
while os.path.exists('%s%d.csv' % (hyp_res_base, cnt)):
    cnt += 1
pd_opt.to_csv('%s%d.csv' % (hyp_res_base, cnt))
