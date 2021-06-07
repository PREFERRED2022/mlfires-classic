#!/usr/bin/env python
from hyperopt import Trials, fmin, tpe, hp, STATUS_OK
import numpy as np
import os
import time
import space_newcv
from functools import partial
import re
from manage_model import run_predict, run_predict_and_metrics, create_model, fit_model
import fileutils
from MLscores import calc_metrics_custom, cmvals, metrics_aggr, \
    metrics_dict, calc_all_model_distrib, metrics_dict_distrib
import sys
from check_and_prepare_dataset import load_dataset
import cv_common

def load_files(cvset, settype, setdir):
    setfiles = []
    for dsfilepattern in cvset[settype]:
        setfiles += [f for f in fileutils.find_files(setdir, dsfilepattern, listtype="walk")]
    return setfiles

def resfilename(opt_target, runmode):
    hyp_res_base = os.path.join('results', 'hyperopt',
                                'hyperopt_results_' +
                                ''.join([ch for ch in opt_target if re.match(r'\w', ch)]) + '_'
                                + filedesc + '_')
    cnt = 1
    while os.path.exists('%s%d.csv' % (hyp_res_base, cnt)):
        cnt += 1
    hyperresfile = '%s%d.csv' % (hyp_res_base, cnt)
    hyperallfile = '%sall_%d.csv' % (hyp_res_base, cnt)
    return hyperresfile, hyperallfile

def evalmodel(cvsets, optimize_target, calc_test, modeltype, hyperresfile, hyperallfile, scoresfile, params):
    if len(cvsets) == 0:
        print('No cross validation files')
        sys.exit()
    metrics = []
    cnt = 0
    print("Params : %s" % params)

    start_cv_all = time.time()
    for cvset in cvsets:
        print('%s Set: %s' % (runmode,cvset))
        trfiles = load_files(cvset, 'training', trainsetdir)
        if len(trfiles) == 0:
            print("No training dataset(s) found")
            return
        print('Training Files: %s' % trfiles)
        X_pd, y_pd, groups_pd = load_dataset(trfiles, params['feature_drop'], debug=debug)
        X_pd = X_pd.reindex(sorted(X_pd.columns), axis=1)
        traincolumns = X_pd.columns

        X_train = X_pd.values
        y_train = y_pd.values
        y_train = y_train[:, 0]
        start_fit = time.time()
        print("Fitting model ...")


        model = create_model(modeltype, params, X_train)
        model, res = fit_model(modeltype, model, params, X_train, y_train)
        '''
        if modeltype == 'tensorflow':
            model = create_NN_model(params, X_train)
            es = EarlyStopping(monitor='loss', patience=10, min_delta=0.002)
            res = model.fit(X_train, y_train, batch_size=512, epochs=params['max_epochs'], verbose=0, callbacks=[es],
                            class_weight=params['class_weights'])
        elif modeltype == 'sklearn':
            model = create_sklearn_model(params, X_train)
            model.fit(X_train, y_train)
        '''
        print("Fit time (min): %.1f" % ((time.time() - start_fit) / 60.0))

        '''training set metrics'''
        start_time_trmetrics = time.time()
        mset = 'train'
        metrics_dict_train, y_scores = run_predict_and_metrics(model, modeltype, X_train, y_train, 'train', not calc_test)
        print("Training metrics time (min): %.1f"%((time.time() - start_time_trmetrics)/60.0))

        if debug:
            calc_metrics_custom(metrics_dict_train['TN %s' % mset], metrics_dict_train['FP %s' % mset],\
                                metrics_dict_train['FN %s' % mset], metrics_dict_train['TP %s' % mset], y_scores, y_train,\
                                numaucthres=numaucthres, debug=debug)
        if modeltype == 'tensorflow':
            es_epochs = len(res.history['loss'])
        else:
            es_epochs = 0

        start_cv = time.time()
        tn = 0; fp = 0; fn = 0; tp = 0;
        y_scores = None; y_pred = None; y_val = None
        cvfiles = load_files(cvset, 'crossval', testsetdir)
        if len(cvfiles) == 0:
            print("No Validation dataset(s) found")
            return
        for cvfile in cvfiles:
            start_predict_file = time.time()
            print('Cross Validation File: %s' % cvfile)
            X_pd, y_pd, groups_pd, id_pd = load_dataset(cvfile, params['feature_drop'], class0nrows=cvrownum,\
                                                 debug=debug, returnid=True)
            X_pd = X_pd.reindex(sorted(X_pd.columns), axis=1)
            valcolumns = X_pd.columns
            if debug:
                for i in range(0, len(traincolumns)):
                    if traincolumns[i] != valcolumns[i]:
                        print('WARNING! Training set column %d: %s is different from Validation Set Column %d: %s' % (
                        i, traincolumns[i], i, valcolumns[i]))

            X_val = X_pd.values
            _y_val = y_pd.values
            _y_val = _y_val[:, 0]
            X_pd = None; y_pd = None; groups_pd = None
            if debug:
                print("Running prediction...")
            _y_scores, _y_pred = run_predict(model, modeltype, X_val)
            if numaucthres>0:
                if y_scores is None:
                    y_scores = _y_scores
                    y_pred = _y_pred
                    y_val = _y_val
                else:
                    y_scores = np.concatenate((y_scores, _y_scores))
                    y_pred = np.concatenate((y_pred, _y_pred))
                    y_val = np.concatenate((y_val, _y_val))
            if debug:
                print("confusion matrix retrieval...")
            _tn, _fp, _fn, _tp = cmvals(_y_val, _y_pred)
            if debug:
                print("file tn : %d, fp : %d, fn : %d, tp : %d" % (_tn, _fp, _fn, _tp))
            tn += _tn;
            fp += _fp;
            fn += _fn;
            tp += _tp;
            if debug:
                print("sums tn : %d, fp : %d, fn : %d, tp : %d" % (tn, fp, fn, tp))
            print("Predict time (min): %.1f" % ((time.time() - start_predict_file) / 60.0))

        '''validation set metrics'''
        msetval = runmode
        metrics_dict_val = metrics_dict(*calc_metrics_custom(tn, fp, fn, tp, y_scores, y_val, \
                                                             numaucthres=numaucthres, debug=debug), msetval)
        metrics_dict_dist = metrics_dict_distrib(*calc_all_model_distrib(y_scores[:, 1], y_val), msetval)
        print("Validation metrics time (min): %.1f" % ((time.time() - start_cv) / 60.0))
        print("Recall 1 val: %.3f, Recall 0 val: %.3f" % (metrics_dict_val['recall 1 %s'%msetval], metrics_dict_val['recall 0 %s'%msetval]))
        metrics_dict_fold = {}
        metrics_dict_fold['fold'] = '%s'%cvset['crossval']
        metrics_dict_fold = {**metrics_dict_fold, **metrics_dict_train, **metrics_dict_val, **metrics_dict_dist}
        if modeltype=='tensorflow':
            metrics_dict_fold['early stop epochs'] = es_epochs
        metrics_dict_fold['params']='%s'%params
        metrics_dict_fold['CV Fit and predict min.']=(time.time() - start_cv)/60.0
        metrics.append(metrics_dict_fold)
        if writescores:
            sfile_suffix=''.join([ch for ch in '%s'%cvset['crossval'] if re.match(r'\w', ch)])
            cv_common.write_score(scoresfile+sfile_suffix+'.csv', None, None, y_val, y_scores[:,1])
    mean_metrics = {}
    mean_metrics = metrics_aggr(metrics, mean_metrics)
    mean_metrics["CV time (min)"] = (time.time() - start_cv_all) / 60.0
    mean_metrics['params'] = '%s'%params
    print('Mean %s : %s' % (optimize_target, mean_metrics[optimize_target]))
    cv_common.writemetrics(metrics, mean_metrics, hyperresfile, hyperallfile)

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

testsets, space, testmodels, max_trials, calc_test, opt_targets, trainsetdir, testsetdir, numaucthres, modeltype, \
cvrownum, filedesc, runmode, writescores, resdir, debug = space_newcv.create_space()
random_state = 42

# tf.config.threading.set_inter_op_parallelism_threads(
#    n_cpus
# )

for opt_target in opt_targets:
    hyperresfile = cv_common.get_filename(opt_target, modeltype, filedesc, aggr='mean', resultsfolder=resdir)
    hyperallfile = cv_common.get_filename(opt_target, modeltype, filedesc, aggr='all', resultsfolder=resdir)
    scoresfile = cv_common.get_filename(opt_target, modeltype, filedesc, aggr='scores', ext='', resultsfolder=resdir)
    if runmode == 'val.':
        trials = Trials()
        evalmodelpart = partial(evalmodel, testsets, opt_target, calc_test, modeltype, \
                                hyperresfile, hyperallfile, scoresfile)
        if not os.path.isdir(os.path.join('results', 'hyperopt')):
            os.makedirs(os.path.join('results', 'hyperopt'))
        best = fmin(fn=evalmodelpart,  # function to optimize
                    space=space,
                    algo=tpe.suggest,  # optimization algorithm, hyperotp will select its parameters automatically
                    max_evals=max_trials,  # maximum number of iterations
                    trials=trials,  # logging
                    rstate=np.random.RandomState(random_state)  # fixing random state for the reproducibility
                    )
    elif runmode == 'test':
        print("Output files : %s, %s"%(hyperresfile,hyperallfile))
        for modelparams in testmodels[opt_target]:
            evalmodel(testsets, opt_target, calc_test, modeltype, hyperresfile, hyperallfile, \
                      scoresfile, modelparams)
