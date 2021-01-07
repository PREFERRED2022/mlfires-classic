#!/usr/bin/env python
from hyperopt import Trials, fmin, tpe, hp, STATUS_OK
from pandas import read_csv
from sklearn.model_selection import train_test_split, KFold, GroupKFold
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
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
from functools import partial
import re

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
    if domdir_col:
        Xbindomdir = pd.get_dummies(X_unnorm[domdir_col].round())
        if 0 in Xbindomdir.columns:
            del Xbindomdir[0]
        ddircols = []
        for i in range(1, 9):
            ddircols.append('bin_dom_dir_%d' % i)
        Xbindomdir.columns = ddircols
        del X_unnorm[domdir_col]
        X_unnorm = pd.concat([X_unnorm, Xbindomdir], axis = 1)

    if dirmax_col:
        Xbindirmax = pd.get_dummies(X_unnorm[dirmax_col].round())
        if 0 in Xbindirmax.columns:
            del Xbindirmax[0]
        dmaxcols = []
        for i in range(1, 9):
            dmaxcols.append('bin_dir_max_%d' % i)
        Xbindirmax.columns = dmaxcols
        del X_unnorm[dirmax_col]
        X_unnorm = pd.concat([X_unnorm, Xbindirmax], axis = 1)

    if corine_col:
        # convert corine level
        corine2 = X_unnorm[corine_col].copy() // 10
        del X_unnorm[corine_col]
        #X_unnorm.rename(columns={corine_col: 'corine_orig'})
        X_unnorm = pd.concat([X_unnorm, corine2], axis=1)

        Xbincorine = pd.get_dummies(X_unnorm[corine_col])
        corcols = ['bin_corine_' + str(c) for c in Xbincorine.columns]
        Xbincorine.columns = corcols
        del X_unnorm[corine_col]
        X_unnorm = pd.concat([X_unnorm, Xbincorine], axis = 1)

    #str_classes = ['Corine']
    #X_unnorm_int = normdataset.index_string_values(X_unnorm, str_classes)
    #X = normdataset.normalize_dataset(X_unnorm_int, 'std')

    X = normdataset.normalize_dataset(X_unnorm)
    y = y_int
    groupspd = df[firedate_col]

    return X, y, groupspd

def check_categorical(df, checkcol, newcols):
    cat_cols = [c for c in df.columns if checkcol.upper() in c.upper()]
    if any([c.upper() == checkcol.upper() for c in cat_cols]) and len(cat_cols) > 1:
        cat_col = [c for c in df.columns if checkcol.upper() == c.upper()][0]
        deletecolumns = []
        for c in newcols:
            if (c.upper() != checkcol.upper() and checkcol.upper() in c.upper()):
                deletecolumns.append(c)
        for c in deletecolumns:
            newcols.remove(c)
    elif any([c.upper() == checkcol.upper() for c in cat_cols]) and len(cat_cols) == 1:
        cat_col = [c for c in df.columns if checkcol.upper() == c.upper()][0]
    elif not any([c.upper() == checkcol.upper() for c in cat_cols]) and len(cat_cols) == 1:
        cat_col = [c for c in df.columns if checkcol.upper() == c.upper()][0]
    elif not any([c.upper() == checkcol.upper() for c in cat_cols]) and len(cat_cols) > 1:
        cat_col = None
        for i in range(0,len(newcols)):
            c = newcols[i]
            if (c.upper() != checkcol.upper() and checkcol.upper() in c.upper()):
                newname = "bin_" + c
                newcols[i] = newname
                df.rename(columns={c : newname}, inplace=True)
    else:
        cat_col = None
    return cat_col, newcols

# load the dataset
def load_dataset():
    dsetfolder = 'data/'
    #dsfile = 'dataset_ndvi_lu.csv'
    domdircheck = 'dom_dir'
    dirmaxcheck = 'dir_max'
    corinecheck = 'Corine'
    firedatecheck = 'firedate'
    X_columns = ['max_temp', 'min_temp', 'mean_temp', 'res_max', dirmaxcheck, 'dom_vel', domdircheck,
                 'rain_7days',
                 corinecheck, 'Slope', 'DEM', 'Curvature', 'Aspect', 'ndvi', 'evi']
    y_columns = ['fire']
    dsreadysuffix = '_nn_ready'
    dsready = dsfile[:-4]+dsreadysuffix+".csv"
    if not os.path.exists(os.path.join(dsetfolder, dsready)):
        df = pd.read_csv(os.path.join(dsetfolder, dsfile))
        X_columns_upper = [c.upper() for c in X_columns]
        newcols = [c for c in df.columns if c.upper() in X_columns_upper or any([cX in c.upper() for cX in X_columns_upper])]
        X_columns = newcols
        corine_col, newcols = check_categorical(df, corinecheck, newcols)
        dirmax_col, newcols = check_categorical(df, dirmaxcheck, newcols)
        domdir_col, newcols = check_categorical(df, domdircheck, newcols)

        firedate_col = [c for c in df.columns if firedatecheck.upper() in c.upper()][0]
        X, y, groupspd = prepare_dataset(df, X_columns, y_columns, firedate_col, corine_col, domdir_col, dirmax_col)
        featdf = pd.concat([X, y, groupspd], axis=1)
        featdf[[c for c in featdf.columns if 'Unnamed' not in c]].to_csv(os.path.join(dsetfolder, dsready))
    else:
        featdf = pd.read_csv(os.path.join(dsetfolder, dsready))
        firedate_col = [c for c in featdf.columns if 'firedate'.upper() in c.upper()][0]
        X_columns_new = [c for c in featdf.columns if c not in [firedate_col,'fire'] and 'Unnamed' not in c]
        X = featdf[X_columns_new]
        y = featdf[y_columns]
        groupspd = featdf[firedate_col]


    #drop_all0_features(featdf)

    return X, y, groupspd


def create_NN_model(params, X):
    # define model
    model = Sequential()
    n_features = X.shape[1]
    intlayers = int(params['n_internal_layers'][0])
    model.add(Dense(params['n_internal_layers'][1]['layer_1_' + str(intlayers) + '_nodes'], activation='relu',
                    input_shape=(n_features,)))
    if params['dropout']:
        model.add(Dropout(0.5))
    for i in range(2, intlayers + 2):
        model.add(Dense(int(params['n_internal_layers'][1]['layer_' + str(i) + '_' + str(intlayers) + '_nodes']),
                        activation='relu'))
        if params['dropout']:
            model.add(Dropout(0.5))

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

def hybridrecall(w1, w0, rec1, rec0):
    if rec1 != 0 and rec0 != 0:
        return (w1+w0) / (w1 / rec1 + w0 / rec0)
    elif rec1 == 0 and rec0 == 0:
        return 0
    elif rec1 == 0:
        return rec0
    elif rec0 == 0:
        return rec1

def calc_metrics(model, X, y, aucmetric, dontcalc = False):
    if dontcalc:
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    y_scores = model.predict(X)
    predict_class = lambda p: int(round(p))
    predict_class_v = np.vectorize(predict_class)
    y_pred = predict_class_v(y_scores[:, 1])

    aucmetric.update_state(y, y_scores[:, 1])
    auc = float(aucmetric.result())

    acc_1 = accuracy_score(y, y_pred)
    acc_0 = accuracy_score(1 - y, 1 - y_pred)

    prec_1 = precision_score(y, y_pred)
    prec_0 = precision_score(1 - y, 1 - y_pred)

    rec_1 = recall_score(y, y_pred)
    rec_0 = recall_score(1 - y, 1 - y_pred)

    f1_1 = f1_score(y, y_pred)
    f1_0 = f1_score(1 - y, 1 - y_pred)

    hybrid1 = hybridrecall(2, 1, rec_1, rec_0)
    hybrid2 = hybridrecall(5, 1, rec_1, rec_0)

    return auc, acc_1, acc_0, prec_1, prec_0, rec_1, rec_0, f1_1, f1_0, hybrid1, hybrid2

#def nnfit(cv=kf, X_pd=X_pd, y_pd=y_pd, groups_pd=groups_pd, params):
def nnfit(cv, X_pd, y_pd, groups_pd, optimize_target, calc_test, params):

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

    if len(params['feature_drop'])>0:
        #X_pd=X_pd.drop(columns=[c for c in X_pd.columns if params['feature_drop'] in c])
        X_pd = X_pd.drop(columns=[c for c in X_pd.columns if any([fd in c for fd in params['feature_drop']])])

    X = X_pd.values
    y = y_pd.values
    groups = groups_pd.values

    start_folds = time.time()
    for train_index, test_index in cv.split(X, y, groups):
        cnt += 1
        print("Fitting Fold %d" % cnt)
        start_fold_time = time.time()
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
        #loss_test, acc_test = model.evaluate(X_val, y_val, batch_size=512, verbose=0)
        #y_pred = model.predict_classes(X_val)
        '''
        y_scores = model.predict(X_val)
        predict_class = lambda p: int(round(p))
        predict_class_v = np.vectorize(predict_class)
        y_pred = predict_class_v(y_scores[:, 1])

        aucmetric.update_state(y_val, y_scores[:,1])
        auc_val = float(aucmetric.result())

        acc_1_test = accuracy_score(y_val, y_pred)
        acc_0_test = accuracy_score(1 - y_val, 1 - y_pred)

        prec_1_test = precision_score(y_val, y_pred)
        prec_0_test = precision_score(1 - y_val, 1 - y_pred)

        rec_1_test = recall_score(y_val, y_pred)
        rec_0_test = recall_score(1 - y_val, 1 - y_pred)

        f1_1_test = f1_score(y_val, y_pred)
        f1_0_test = f1_score(1 - y_val, 1 - y_pred)

        hybrid1_test = hybridrecall(2, 1, rec_1_test, rec_0_test)
        hybrid2_test = hybridrecall(5, 1, rec_1_test, rec_0_test)
        '''

        auc_val,acc_1_test,acc_0_test,prec_1_test, prec_0_test,rec_1_test, rec_0_test,f1_1_test,f1_0_test,hybrid1_test,hybrid2_test \
            = calc_metrics(model, X_val, y_val, aucmetric, False)

        print("Validation metrics time (min): %s"%((time.time() - start_time)/60.0))
        start_time = time.time()

        '''training set metrics'''

        #loss_train, acc_train = model.evaluate(X_train, y_train, batch_size=512, verbose=0)
        #y_pred = model.predict_classes(X_train)
        '''
        y_scores = model.predict(X_train)
        y_pred = predict_class_v(y_scores[:, 1])

        aucmetric.update_state(y_train, y_scores[:,1])
        auc_train = float(aucmetric.result())

        acc_1_train = accuracy_score(y_train, y_pred)
        acc_0_train = accuracy_score(1-y_train, 1-y_pred)

        prec_1_train = precision_score(y_train, y_pred)
        prec_0_train = precision_score(1-y_train, 1-y_pred)

        rec_1_train = recall_score(y_train, y_pred)
        rec_0_train = recall_score(1-y_train, 1-y_pred)

        f1_1_train = f1_score(y_train, y_pred)
        f1_0_train = f1_score(1-y_train, 1-y_pred)

        hybrid1_train = hybridrecall(2, 1, rec_1_train, rec_0_train)
        hybrid2_train = hybridrecall(5, 1, rec_1_train, rec_0_train)
        '''

        auc_train,acc_1_train,acc_0_train,prec_1_train, prec_0_train,rec_1_train, rec_0_train,f1_1_train,f1_0_train,hybrid1_train,hybrid2_train \
            = calc_metrics(model, X_train, y_train, aucmetric, not calc_test)


        print("Training metrics time (min): %s"%((time.time() - start_time)/60.0))
        print("Recall 1 val: %s, Recall 0 val: %s" % (rec_1_test,rec_0_test))

        metrics.append(
            {#'loss val.': loss_test, 'loss train': loss_train,
             'accuracy val.': acc_1_test, 'accuracy train': acc_1_train,
             'precision 1 val.': prec_1_test, 'precision 1 train': prec_1_train, 'recall 1 val.' : rec_1_test,
             'recall 1 train': rec_1_train,'f1-score 1 val.': f1_1_test, 'f1-score 1 train': f1_1_train,
             'accuracy 0 val.': acc_0_test, 'accuracy 0 train': acc_0_train,
             'precision 0 val.': prec_0_test, 'precision 0 train': prec_0_train, 'recall 0 val.': rec_0_test,
             'recall 0 train': rec_0_train, 'f1-score 0 val.': f1_0_test, 'f1-score 0 train': f1_0_train,
             'auc val.': auc_val,
             'auc train.': auc_train, 'hybrid1 train': hybrid1_train, 'hybrid1 val': hybrid1_test, 'hybrid2 train': hybrid2_train, 'hybrid2 val': hybrid2_test,
             'early stop epochs': es_epochs
             } )#'fit time':  (time.time() - start_fold_time)/60.0})

        #print(metrics[-1])

    mean_metrics = {}
    for m in metrics[0]:
        mean_metrics[m] = sum(item.get(m, 0) for item in metrics) / len(metrics)
    mean_metrics["fit time (min)"] = (time.time() - start_folds)/60.0
    print('Mean %s : %s' % (optimize_target,mean_metrics[optimize_target]))

    return {
        'loss': -mean_metrics[optimize_target],
        'status': STATUS_OK,
        # -- store other results like this
        # 'eval_time': time.time(),
        'metrics': mean_metrics,
        'params': '%s'%params
        # -- attachments are handled differently
        # 'attachments':
        #    {'time_module': pickle.dumps(time.time)}
    }


tset, testsets, space, max_trials, max_epochs, calc_test, opt_targets = space.create_space()
dsfile = testsets[tset]
X_pd, y_pd, groups_pd = load_dataset()

opt_targets = ['hybrid1 val', 'hybrid2 val', 'f1-score 1 val.', 'auc val.', 'recall 1 val.']
for opt_target in opt_targets:
    trials = Trials()
    nnfitpart = partial(nnfit, kf, X_pd, y_pd, groups_pd, opt_target, calc_test)

    best = fmin(fn=nnfitpart,  # function to optimize
                space=space,
                algo=tpe.suggest,  # optimization algorithm, hyperotp will select its parameters automatically
                max_evals=max_trials,  # maximum number of iterations
                trials=trials,  # logging
                rstate=np.random.RandomState(random_state)  # fixing random state for the reproducibility
                )

    pd_opt = pd.DataFrame(columns=list(trials.trials[0]['result']['metrics'].keys()))
    for t in trials:
        pdrow = t['result']['metrics']
        pdrow['params'] = t['result']['params']
        pd_opt = pd_opt.append(pdrow, ignore_index=True)

    if not os.path.isdir(os.path.join('results','hyperopt')):
        os.makedirs(os.path.join('results','hyperopt'))

    hyp_res_base = os.path.join('results','hyperopt','hyperopt_results_'+"".join([ch for ch in opt_target if re.match(r'\w', ch)])+'_'+tset+'_')
    cnt = 1
    while os.path.exists('%s%d.csv' % (hyp_res_base, cnt)):
        cnt += 1
    pd_opt.to_csv('%s%d.csv' % (hyp_res_base, cnt))
