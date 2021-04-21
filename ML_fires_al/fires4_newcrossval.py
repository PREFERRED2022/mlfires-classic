#!/usr/bin/env python
from hyperopt import Trials, fmin, tpe, hp, STATUS_OK
from pandas import read_csv
from sklearn.model_selection import train_test_split, KFold, GroupKFold
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
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
import space_newcv
from functools import partial
import re
import manage_model
import fileutils
import MLscores
import sys
from csv import DictWriter

num_folds = 10
# kf = KFold(n_splits=num_folds, shuffle=True)
kf = GroupKFold(n_splits=num_folds)
random_state = 42


def drop_all0_features(df):
    for c in df.columns:
        if 'bin' in c:
            u = df[c].unique()
            if (len(u) > 1):
                cnt = df.groupby([c]).count()
                if cnt["max_temp"][1] < 20:
                    print("Droping Feature %s, count: %d" % (c, cnt["max_temp"][1]))
                    df.drop(columns=[c])
            else:
                print("%s not exists (size : %d)" % (c, len(u)))


def prepare_dataset(df, X_columns, y_columns, firedate_col, corine_col, domdir_col, dirmax_col):
    df = df.dropna()
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
        X_unnorm = pd.concat([X_unnorm, Xbindomdir], axis=1)

    if dirmax_col:
        Xbindirmax = pd.get_dummies(X_unnorm[dirmax_col].round())
        if 0 in Xbindirmax.columns:
            del Xbindirmax[0]
        dmaxcols = []
        for i in range(1, 9):
            dmaxcols.append('bin_dir_max_%d' % i)
        Xbindirmax.columns = dmaxcols
        del X_unnorm[dirmax_col]
        X_unnorm = pd.concat([X_unnorm, Xbindirmax], axis=1)

    if corine_col:
        # convert corine level
        corine2 = X_unnorm[corine_col].copy() // 10
        del X_unnorm[corine_col]
        # X_unnorm.rename(columns={corine_col: 'corine_orig'})
        X_unnorm = pd.concat([X_unnorm, corine2], axis=1)

        Xbincorine = pd.get_dummies(X_unnorm[corine_col])
        corcols = ['bin_corine_' + str(c) for c in Xbincorine.columns]
        Xbincorine.columns = corcols
        del X_unnorm[corine_col]
        X_unnorm = pd.concat([X_unnorm, Xbincorine], axis=1)

    # X = normdataset.normalize_dataset(X_unnorm, aggrfile='stats/featurestats.json')
    X = X_unnorm
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
        for i in range(0, len(newcols)):
            c = newcols[i]
            if (c.upper() != checkcol.upper() and checkcol.upper() in c.upper()):
                newname = "bin_" + c
                newcols[i] = newname
                df.rename(columns={c: newname}, inplace=True)
    else:
        cat_col = None
    return cat_col, newcols

# load the dataset
def load_dataset(trfiles, featuredrop=None, class0nrows=0):
    # dsfile = 'dataset_ndvi_lu.csv'
    domdircheck = 'dom_dir'
    dirmaxcheck = 'dir_max'
    corinecheck = 'Corine'
    monthcheck = 'month'
    wkdcheck = 'wkd'
    firedatecheck = 'firedate'
    X_columns = ['max_temp', 'min_temp', 'mean_temp', 'res_max', dirmaxcheck, 'dom_vel', domdircheck,
                 'rain_7days', corinecheck, 'Slope', 'DEM', 'Curvature', 'Aspect', 'ndvi', 'evi', 'lst_day',
                 'lst_night', monthcheck, wkdcheck,
                 'mean_dew_temp', 'max_dew_temp', 'min_dew_temp','frequency', 'f81', 'x', 'y']
    y_columns = ['fire']
    # if not os.path.exists(os.path.join(dsetfolder, dsready)):
    if isinstance(trfiles, list):
        dflist=[]
        for dsfile in trfiles:
            dflist.append(pd.read_csv(dsfile))
        df=pd.concat(dflist)
    else:
        dsfile = trfiles
        if class0nrows > 0:
            dffirefile = dsfile[0:-4]+"_fires.csv"
            if os.path.isfile(dffirefile):
                if debug:
                    print("Loading fire dataset %s"%dffirefile)
                dffire = pd.read_csv(dffirefile)
                if debug:
                    print("Loading no-fire dataset %s"%dffirefile)
                dfpart = pd.read_csv(dsfile, nrows=class0nrows)
                dfpart = dfpart[dfpart['fire']!=1]
                df = pd.concat([dfpart, dffire])
            else:
                df = pd.read_csv(dsfile)
                if debug:
                    print("Split dataset to fire no-fire")
                firegroup = df.groupby('fire')
                dfpart = firegroup.get_group(0).head(class0nrows)
                if 1 in firegroup.groups:
                    if debug:
                        print("Creating fire dataset %s" % dffirefile)
                    firegroup.get_group(1).to_csv(dffirefile, index=False)
                    df = pd.concat([dfpart, firegroup.get_group(1)])
                else:
                    df = dfpart
        else:
            df = pd.read_csv(dsfile)

    X_columns_upper = [c.upper() for c in X_columns]
    newcols = [c for c in df.columns if
               c.upper() in X_columns_upper or any([cX in c.upper() for cX in X_columns_upper])]
    X_columns = newcols
    corine_col, newcols = check_categorical(df, corinecheck, newcols)
    dirmax_col, newcols = check_categorical(df, dirmaxcheck, newcols)
    domdir_col, newcols = check_categorical(df, domdircheck, newcols)
    month_col, newcols = check_categorical(df, monthcheck, newcols)
    wkd_col, newcols = check_categorical(df, wkdcheck, newcols)

    firedate_col = [c for c in df.columns if firedatecheck.upper() in c.upper()][0]
    X, y, groupspd = prepare_dataset(df, X_columns, y_columns, firedate_col, corine_col, domdir_col, dirmax_col)
    print("Ignored columns from csv %s"%([c for c in df.columns if c not in X.columns]))
    X_columns = X.columns
    if len(featuredrop) > 0:
        X = X.drop(columns=[c for c in X.columns if any([fd in c for fd in featuredrop])])
    print("Dropped columns %s"%(list(set(X_columns)-set(X.columns))))
    #if debug:
    #    print("X helth check %s"%X.describe())
    #    print("y helth check %s"%y.describe())
    return X, y, groupspd


def hybridrecall(w1, w0, rec1, rec0):
    if rec1 > 0 and rec0 > 0:
        return (w1 + w0) / (w1 / rec1 + w0 / rec0)
    else:
        return -1000

def calc_metrics(y, y_scores, y_pred):
    if debug:
        print("calulating merics from scores (sklearn)")
        print("calulating tn, fp, fn, tp")
    tn, fp, fn, tp = MLscores.cmvals(y, y_pred)
    if debug:
        print("tn : %d, fp : %d, fn : %d, tp : %d"%(tn, fp, fn, tp))
    if debug:
        print("calulating auc...")
    # aucmetric = tensorflow.metrics.AUC(num_thresholds=numaucthres)
    # aucmetric.update_state(y, y_scores[:, 1])
    # auc = float(aucmetric.result())
    auc = 0.0
    if debug:
        print("auc : %.2f"%auc)
    if debug:
        print("calulating accuracy...")
    acc_1 = accuracy_score(y, y_pred)
    acc_0 = accuracy_score(1 - y, 1 - y_pred)
    if debug:
        print("accuracy 1 : %.2f"%acc_1)
        print("accuracy 0 : %.2f"%acc_0)
    if debug:
        print("calulating recall...")
    rec_1 = recall_score(y, y_pred)
    rec_0 = recall_score(1 - y, 1 - y_pred)
    if debug:
        print("recall 1 : %.2f"%rec_1)
        print("recall 0 : %.2f"%rec_0)
    if debug:
        print("calulating precision...")
    prec_1 = precision_score(y, y_pred)
    prec_0 = precision_score(1 - y, 1 - y_pred)
    if debug:
        print("precision 1 : %.2f"%prec_1)
        print("precision 0 : %.2f"%prec_0)
    if debug:
        print("calulating f1 score...")
    f1_1 = f1_score(y, y_pred)
    f1_0 = f1_score(1 - y, 1 - y_pred)
    if debug:
        print("f1 1 : %.2f"%f1_1)
        print("f1 0 : %.2f"%f1_0)
    if debug:
        print("calulating hybrids...")
    hybrid1 = hybridrecall(2, 1, rec_1, rec_0)
    hybrid2 = hybridrecall(5, 1, rec_1, rec_0)
    if debug:
        print("hybrid 1 : %.2f"%hybrid1)
        print("hybrid 2 : %.2f"%hybrid2)
    # tp0 = tn1 tn0 = tp1 fp0 = fn1 fn0 = fp1
    return auc, acc_1, acc_0, prec_1, prec_0, rec_1, rec_0, f1_1, f1_0, hybrid1, hybrid2, tn, fp, fn, tp


def calc_metrics_custom(tn, fp, fn, tp):
    if debug:
        print("calulating merics (custom)")
    if debug:
        print("(input) tn : %d, fp : %d, fn : %d, tp : %d"%(tn, fp, fn, tp))
    auc = 0
    if debug:
        print("auc : %.2f"%auc)
    ##############################################
    # tp0 = tn1, tn0 = tp1, fp0 = fn1, fn0 = fp1 #
    ##############################################
    if debug:
        print("calulating accuracy...")
    acc_1 = MLscores.accuracy(tp, tn, fp, fn)
    acc_0 = MLscores.accuracy(tn, tp, fn, fp)
    if debug:
        print("accuracy 1 : %.2f"%acc_1)
        print("accuracy 0 : %.2f"%acc_0)
    if debug:
        print("calulating recall ...")
    rec_1 = MLscores.recall(tp, fn)
    rec_0 = MLscores.recall(tn, fp)
    if debug:
        print("recall 1 : %.2f"%rec_1)
        print("recall 0 : %.2f"%rec_0)
    if debug:
        print("calulating precision...")
    prec_1 = MLscores.precision(tp, fp)
    prec_0 = MLscores.precision(tn, fn)
    if debug:
        print("precision 1 : %.2f"%prec_1)
        print("precision 0 : %.2f"%prec_0)
    if debug:
        print("calulating f1_score...")
    f1_1 = MLscores.f1(tp, fp, fn)
    f1_0 = MLscores.f1(tn, fn, fp)
    if debug:
        print("f1 1 : %.2f" % f1_1)
        print("f1 0 : %.2f" % f1_0)
    if debug:
        print("calulating hybrids ...")
    hybrid1 = hybridrecall(2, 1, rec_1, rec_0)
    hybrid2 = hybridrecall(5, 1, rec_1, rec_0)
    if debug:
        print("hybrid 1 : %.2f"%hybrid1)
        print("hybrid 2 : %.2f"%hybrid2)
    return auc, acc_1, acc_0, prec_1, prec_0, rec_1, rec_0, f1_1, f1_0, hybrid1, hybrid2, tn, fp, fn, tp

def run_predict(model, X):
    if debug:
        print("Running Prediction...")
    if modeltype=='tensorflow':
        y_scores = model.predict(X)
    elif modeltype=='sklearn':
        y_scores = model.predict_proba(X)
    predict_class = lambda p: int(round(p))
    predict_class_v = np.vectorize(predict_class)
    y_pred = predict_class_v(y_scores[:, 1])
    return y_scores, y_pred

def run_predict_and_metrics(model, X, y, dontcalc=False):
    if dontcalc:
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    y_scores, y_pred = run_predict(model, X)
    return calc_metrics(y, y_scores, y_pred)

def load_files(cvset, settype, setdir):
    setfiles = []
    for dsfilepattern in cvset[settype]:
        setfiles += [f for f in fileutils.find_files(setdir, dsfilepattern, listtype="walk")]
    return setfiles

def writemetrics(metrics, mean_metrics, hpresfile, allresfile):
    writeheader = True if not os.path.isfile(hpresfile) else False
    with open(hpresfile, 'a') as _f:
        dw = DictWriter(_f, fieldnames=mean_metrics.keys())
        if writeheader:
            dw.writeheader()
        dw.writerow(mean_metrics)
    writeheader = True if not os.path.isfile(allresfile) else False
    with open(allresfile, 'a') as _f:
        dw = DictWriter(_f, fieldnames=metrics[0].keys())
        if writeheader:
            dw.writeheader()
        for m in metrics:
            dw.writerow(m)

def resfilename(opt_target):
    hyp_res_base = os.path.join('results', 'hyperopt',
                                'hyperopt_results_' + "".join([ch for ch in opt_target if re.match(r'\w', ch)]) + '_' + filedesc + '_')
    cnt = 1
    while os.path.exists('%s%d.csv' % (hyp_res_base, cnt)):
        cnt += 1
    hyperresfile='%s%d.csv' % (hyp_res_base, cnt)
    hyperallfile='%sall_%d.csv' % (hyp_res_base, cnt)
    return hyperresfile, hyperallfile

# def nnfit(cv=kf, X_pd=X_pd, y_pd=y_pd, groups_pd=groups_pd, params):
def evalmodel(cvsets, optimize_target, calc_test, modeltype, hyperresfile, hyperallfile, params):
    # the function gets a set of variable parameters in "param"
    '''
    params = {'n_internal_layers': params['n_internal_layers'][0],
              'layer_1_nodes': params['n_internal_layers'][1],
              'layer_2_nodes': params['layer_2_nodes'],
              'layer_3_nodes': params['layer_3_nodes'],
              'layer_4_nodes': params['layer_4_nodes']}
              '''

    if len(cvsets) == 0:
        print('No cross validation files')
        sys.exit()
    metrics = []
    cnt = 0
    print("Params : %s" % params)

    start_cv_all = time.time()
    for cvset in cvsets:
        print('Cross Validation Set: %s' % cvset)
        trfiles = load_files(cvset, 'training', trainsetdir)
        print('Training Files: %s' % trfiles)
        X_pd, y_pd, groups_pd = load_dataset(trfiles, params['feature_drop'])
        traincolumns = X_pd.columns

        X_train = X_pd.values
        y_train = y_pd.values
        y_train = y_train[:, 0]
        start_fit = time.time()

        if modeltype=='tensorflow':
            model = manage_model.create_NN_model(params, X_train)
            es = EarlyStopping(monitor='loss', patience=10, min_delta=0.002)
            res = model.fit(X_train, y_train, batch_size=512, epochs=params['max_epochs'], verbose=0, callbacks=[es],
                        class_weight=params['class_weights'])
        elif modeltype=='sklearn':
            model = manage_model.create_sklearn_model(params, X_train)
            model.fit(X_train, y_train)

        print("Fit time (min): %.1f" % ((time.time() - start_fit) / 60.0))

        '''training set metrics'''
        auc_train, acc_1_train, acc_0_train, prec_1_train, prec_0_train, rec_1_train, rec_0_train, f1_1_train, f1_0_train, hybrid1_train, hybrid2_train,\
        tn_train, fp_train, fn_train, tp_train = run_predict_and_metrics(model, X_train, y_train, not calc_test)
        if debug:
            calc_metrics_custom(tn_train, fp_train, fn_train, tp_train)
        if modeltype=='tensorflow':
            es_epochs = len(res.history['loss'])
        else:
            es_epochs = 0

        start_cv = time.time()
        tn = 0; fp = 0; fn = 0; tp = 0;
        cvfiles = load_files(cvset, 'crossval', testsetdir)
        for cvfile in cvfiles:
            start_predict_file = time.time()
            print('Cross Validation File: %s' % cvfile)
            X_pd, y_pd, groups_pd = load_dataset(cvfile, params['feature_drop'], cvrownum)
            valcolumns = X_pd.columns
            if debug:
                for i in range(0,len(traincolumns)):
                    if traincolumns[i]!=valcolumns[i]:
                        print('WARNING! Training set column %d: %s is different from Validation Set Column %d: %s'%(i,traincolumns[i],i,valcolumns[i]))

            X_val = X_pd.values
            _y_val = y_pd.values
            _y_val = _y_val[:, 0]
            _y_scores, _y_pred = run_predict(model, X_val)
            '''
            if y_scores is None:
                y_scores = _y_scores
                y_pred = _y_pred
                y_val = _y_val
            else:
                y_scores = np.concatenate((y_scores, _y_scores))
                y_pred = np.concatenate((y_pred, _y_pred))
                y_val = np.concatenate((y_val, _y_val))
            '''
            if debug:
                print("confusion matrix retrieval...")
            _tn, _fp, _fn, _tp = MLscores.cmvals(_y_val, _y_pred)
            if debug:
                print("file tn : %d, fp : %d, fn : %d, tp : %d"%(_tn, _fp, _fn, _tp))
            tn += _tn; fp += _fp; fn += _fn; tp += _tp;
            if debug:
                print("sums tn : %d, fp : %d, fn : %d, tp : %d"%(tn, fp, fn, tp))
            print("Predict time (min): %.1f" % ((time.time() - start_predict_file) / 60.0))

        '''validation set metrics'''
        auc_val, acc_1_val, acc_0_val, prec_1_val, prec_0_val, rec_1_val, rec_0_val, f1_1_val, f1_0_val, hybrid1_val, hybrid2_val,\
        tn_val, fp_val, fn_val, tp_val = calc_metrics_custom(tn, fp, fn, tp)

        print("Validation metrics time (min): %.1f" % ((time.time() - start_cv) / 60.0))
        start_time = time.time()

        print("Recall 1 val: %s, Recall 0 val: %s" % (rec_1_val, rec_0_val))
        metrics.append(
            {
             'training set': '%s' % cvset['training'],
             '%s set'%valst : '%s' % cvset['crossval'],
             'accuracy %s'%valst: acc_1_val, 'accuracy train': acc_1_train,
             'precision 1 %s'%valst: prec_1_val, 'precision 1 train': prec_1_train, 'recall 1 %s'%valst: rec_1_val,
             'recall 1 train': rec_1_train, 'f1-score 1 %s'%valst: f1_1_val, 'f1-score 1 train': f1_1_train,
             'accuracy 0 %s'%valst: acc_0_val, 'accuracy 0 train': acc_0_train,
             'precision 0 %s'%valst: prec_0_val, 'precision 0 train': prec_0_train, 'recall 0 %s'%valst: rec_0_val,
             'recall 0 train': rec_0_train, 'f1-score 0 %s'%valst: f1_0_val, 'f1-score 0 train': f1_0_train,
             'auc %s'%valst: auc_val,
             'auc train.': auc_train, 'hybrid1 train': hybrid1_train, 'hybrid1 val': hybrid1_val,
             'hybrid2 train': hybrid2_train, 'hybrid2 val': hybrid2_val,
             'TN %s'%valst: tn_val, 'FP %s'%valst: fp_val, 'FN %s'%valst: fn_val, 'TP %s'%valst: tp_val,
             'TN train.': tn_train, 'FP train.': fp_train, 'FN train.': fn_train,  'TP train.': tp_train,
             'early stop epochs': es_epochs,
             'params':'%s'%params
             })  # 'fit time':  (time.time() - start_fold_time)/60.0})
    mean_metrics = {}
    for m in metrics[0]:
        if isinstance(metrics[0][m], str):
            continue
        metricsum = sum([item.get(m, 0) for item in metrics if item.get(m) >= 0])
        cmvalsts = ['TN','FP','FN','TP']
        if any([st in m for st in cmvalsts]):
            mean_metrics[m] = metricsum
        else:
            mean_metrics[m] = metricsum / len(metrics)
    mean_metrics["CV time (min)"] = (time.time() - start_cv_all) / 60.0
    mean_metrics['params'] = '%s'%params
    print('Mean %s : %s' % (optimize_target, mean_metrics[optimize_target]))
    writemetrics(metrics, mean_metrics, hyperresfile, hyperallfile)

    return {
        'loss': -mean_metrics[optimize_target],
        'status': STATUS_OK,
        # -- store other results like this
        # 'eval_time': time.time(),
        'metrics': mean_metrics,
        'params': '%s' % params,
        'allmetrics': metrics,
        # -- attachments are handled differently
        # 'attachments':
        #    {'time_module': pickle.dumps(time.time)}
    }

testsets, space, max_trials, calc_test, opt_targets, n_cpus, trainsetdir, testsetdir, numaucthres, modeltype,\
cvrownum, filedesc, valst, debug = space_newcv.create_space()
tf.config.threading.set_inter_op_parallelism_threads(
    n_cpus
)

for opt_target in opt_targets:

    trials = Trials()
    hyperresfile, hyperallfile = resfilename(opt_target)
    evalmodelpart = partial(evalmodel, testsets, opt_target, calc_test, modeltype, hyperresfile, hyperallfile)

    if not os.path.isdir(os.path.join('results', 'hyperopt')):
        os.makedirs(os.path.join('results', 'hyperopt'))
    best = fmin(fn=evalmodelpart,  # function to optimize
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

    #pd_opt.to_csv(hyperresfile, index=False)
