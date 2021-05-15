#!/usr/bin/env python
from hyperopt import Trials, fmin, tpe, hp, STATUS_OK
from sklearn.model_selection import train_test_split, KFold, GroupKFold
import tensorflow.keras.metrics
import numpy as np
import pandas as pd
import normdataset
import os
import re
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
import time
import space
from functools import partial
from manage_model import create_model, run_predict_and_metrics, run_predict, fit_model
import cv_common
from MLscores import calc_metrics, calc_metrics_custom, cmvals, metrics_aggr

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

    X = normdataset.normalize_dataset(X_unnorm, aggrfile='stats/featurestats.json')
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
        featdf = pd.read_csv(os.path.join(dsetfolder, dsready), dtype={'firedate':str})
        featdf.rename(columns={'x': 'xpos', 'y':'ypos'}, inplace=True)
        featdf = featdf[[c for c in featdf.columns if 'Unnamed' not in c]]
        print('before nan drop: %d'%len(featdf.index))
        featdf = featdf.dropna()
        print('after nan drop: %d' % len(featdf.index))
        firedate_col = [c for c in featdf.columns if 'firedate'.upper() in c.upper()][0]
        X_columns_new = [c for c in featdf.columns if c not in [firedate_col,'fire','id'] and 'Unnamed' not in c]
        X = featdf[X_columns_new]
        y = featdf[y_columns]
        groupspd = featdf[firedate_col]

    #drop_all0_features(featdf)

    return X, y, groupspd
'''
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
    y_scores, y_pred = manage_model.run_predict(model, modeltype, X)

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
'''

def get_filename(opt_target, modeltype, desc, aggr='mean'):
    base_name = os.path.join('results', 'hyperopt', 'hyperopt_results_'+ modeltype + '_' + desc + '_'+ aggr+'_'+"".join([ch for ch in opt_target if re.match(r'\w', ch)]) + '_')
    cnt = 1
    while os.path.exists('%s%d.csv' % (base_name, cnt)):
        cnt += 1
    return '%s%d.csv' % (base_name, cnt)

def validatemodel(cv, X_pd, y_pd, groups_pd, optimize_target, calc_test, modeltype, hpresfile, allresfile, params):

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
        dropcols = [c for c in X_pd.columns if any([fd in c for fd in params['feature_drop']])]
        print("dropping columns: %s"%dropcols)
        X_pd = X_pd.drop(columns=dropcols)

    X = X_pd.values
    y = y_pd.values
    groups = groups_pd.values

    start_folds = time.time()
    for train_index, test_index in cv.split(X, y, groups):
        cnt += 1
        '''Fitting'''
        print("Fitting Fold %d" % cnt)
        start_fold_time = time.time()
        X_train, X_val = X[train_index], X[test_index]
        y_train, y_val = y[train_index], y[test_index]
        y_val = y_val[:,0]
        y_train = y_train[:,0]
        start_time = time.time()
        es_epochs = 0
        model = create_model(modeltype, params, X_train)
        model, res = fit_model(modeltype, model, params, X_train, y_train, X_val, y_val)
        if modeltype == 'tensorflow':
            es_epochs = len(res.history['loss'])

        print("Fit time (min): %.1f"%((time.time() - start_time)/60.0))
        start_time = time.time()

        '''training set metrics'''
        metrics_dict_train, y_scores = run_predict_and_metrics(model, modeltype, X_train, y_train, 'train', not calc_test)
        print("Training metrics time (min): %.1f"%((time.time() - start_time)/60.0))
        start_time = time.time()

        '''validation set metrics'''
        mset = 'val.'
        metrics_dict_val, y_scores = run_predict_and_metrics(model, modeltype, X_val, y_val, mset)

        print("Validation metrics time (min): %.1f"%((time.time() - start_time)/60.0))
        print("Recall 1 val: %.3f, Recall 0 val: %.3f" % (metrics_dict_val['recall 1 %s'%mset],metrics_dict_val['recall 0 %s'%mset]))

        metrics_dict_fold = {}
        metrics_dict_fold['fold'] = 'fold %d'%cnt
        metrics_dict_fold = {**metrics_dict_fold, **metrics_dict_train, **metrics_dict_val}
        if modeltype=='tensorflow':
            metrics_dict_fold['early stop epochs'] = es_epochs
        metrics_dict_fold['params']='%s'%params
        metrics_dict_fold['CV fold fit time']=(time.time() - start_fold_time)/60.0
        metrics.append(metrics_dict_fold)

    mean_metrics = {}
    mean_metrics = metrics_aggr(metrics, mean_metrics)
    mean_metrics["CV time (min)"] = (time.time() - start_folds)/60.0
    mean_metrics['params'] = '%s' % params

    print('Mean %s : %.4f' % (optimize_target,mean_metrics[optimize_target]))
    cv_common.writemetrics(metrics, mean_metrics, hpresfile, allresfile)

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

random_state = 42
tset, testsets, num_folds, space, max_trials, calc_test, opt_targets, modeltype, desc = space.create_space()
kf = GroupKFold(n_splits=num_folds)
#tf.config.threading.set_inter_op_parallelism_threads(
#    n_cpus
#)
dsfile = testsets[tset]
X_pd, y_pd, groups_pd = load_dataset()

for opt_target in opt_targets:
    hpresfile = get_filename(opt_target, modeltype, desc, aggr='mean')
    allresfile = get_filename(opt_target, modeltype, desc, aggr='all')
    trials = Trials()
    validatemodelpart = partial(validatemodel, kf, X_pd, y_pd, groups_pd, opt_target, calc_test, modeltype, hpresfile, allresfile)

    best = fmin(fn=validatemodelpart,  # function to optimize
                space=space,
                algo=tpe.suggest,  # optimization algorithm, hyperotp will select its parameters automatically
                max_evals=max_trials,  # maximum number of iterations
                trials=trials,  # logging
                rstate=np.random.RandomState(random_state)  # fixing random state for the reproducibility
                )
    '''
    pd_opt = pd.DataFrame(columns=list(trials.trials[0]['result']['metrics'].keys()))
    for t in trials:
        pdrow = t['result']['metrics']
        pdrow['params'] = t['result']['params']
        pd_opt = pd_opt.append(pdrow, ignore_index=True)

    if not os.path.isdir(os.path.join('results','hyperopt')):
        os.makedirs(os.path.join('results','hyperopt'))

    hyp_res_base = os.path.join('results','hyperopt','hyperopt_results_'+"".join([ch for ch in opt_target if re.match(r'\w', ch)])+'_'+tset+'_'+modeltype+'_')
    cnt = 1
    while os.path.exists('%s%d.csv' % (hyp_res_base, cnt)):
        cnt += 1
    pd_opt.to_csv('%s%d.csv' % (hyp_res_base, cnt), index=False)
    '''
