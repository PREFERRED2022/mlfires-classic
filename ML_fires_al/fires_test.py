#from hyperopt import Trials, fmin, tpe, STATUS_OK
import numpy as np
import os
import time
import space_new_test
import re
from manage_model import run_predict_q, run_predict_and_metrics_q, create_and_fit_q, run_predict, \
    run_predict_and_metrics, fit_model, create_model, allowgrowthgpus
import fileutils
from MLscores import calc_metrics_custom, cmvals, metrics_aggr, metrics_aggr2, \
    metrics_dict, calc_all_model_distrib, metrics_dict_distrib
import sys
from check_and_prepare_dataset import load_dataset
import cv_common
import best_models
import multiprocessing as mp

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

def runmlprocess(target, args, varnum=0):
    if varnum > 0:
        q = mp.Queue()
        args=tuple(args+[q])
    proc = mp.Process(target=target, args=args)
    proc.start()
    res=[]
    for v in range(varnum):
        res.append(q.get())
    proc.join()
    res=tuple(res)
    return res

def training(cvset, calc_test, modeltype, numaucthres, optimize_target, numtrains, params):
    trfiles = load_files(cvset, 'training', trainsetdir)
    if len(trfiles) == 0:
        print("No training dataset(s) found")
        return
    print('Training Files: %s' % trfiles)
    bestmodel = None
    prevmax = None

    allowgrowthgpus()
    for i in range(numtrains):
        X_pd, y_pd, groups_pd = load_dataset(trfiles, params['feature_drop'], debug=debug)
        X_pd = X_pd.reindex(sorted(X_pd.columns), axis=1)
        traincolumns = X_pd.columns

        X_train = X_pd.values
        y_train = y_pd.values
        y_train = y_train[:, 0]
        start_fit = time.time()
        print("Fitting model ...")

        model = create_model(modeltype, params, X_train)
        model, res = fit_model(modeltype, model, params, X_train, y_train, X_train, y_train)

        # res = runmlprocess(create_and_fit_q, [modeltype, params, X_train, y_train, X_train, y_train], 1)

        tsecs = time.time() - start_fit
        print("Fit time : %d min %d secs" % (int(tsecs/60.0), tsecs % 60))

        '''training set metrics'''
        start_time_trmetrics = time.time()
        mset = 'train'
        metrics_dict_train, y_scores = run_predict_and_metrics(model, modeltype, X_train, y_train, 'train', not calc_test)

        # metrics_dict_train, y_scores = runmlprocess(run_predict_and_metrics_q, [modeltype, X_train, y_train, 'train',
        #                                                       not calc_test, numaucthres],2)
        tsecs = time.time() - start_time_trmetrics
        print("Training metrics time : %d min %d secs" % (int(tsecs/60.0), tsecs % 60))

        if debug:
            calc_metrics_custom(metrics_dict_train['TN %s' % mset], metrics_dict_train['FP %s' % mset], \
                                metrics_dict_train['FN %s' % mset], metrics_dict_train['TP %s' % mset], y_scores, y_train, \
                                numaucthres=numaucthres, debug=debug)
        if modeltype == 'tf':
            es_epochs = len(res.history['loss'])
        else:
            es_epochs = 0

        all_metrics = metrics_aggr([metrics_dict_train], {})
        trtarget = optimize_target.replace('test','train')
        if prevmax is None or all_metrics[trtarget] > prevmax:
            bestmodel = model
            bestmetrics = metrics_dict_train
            bestepochs = es_epochs
            bestiter=i
            prevmax = all_metrics[trtarget]
        print("Training Iteration #%d/%d. BEST iteration so far (#%d) for %s : %.3f" %
              (i+1, numtrains, bestiter+1, optimize_target, prevmax))
    return traincolumns, bestmodel, bestmetrics, bestepochs


def evalmodel(cvsets, optimize_target, calc_test, modeltype, hyperresfile, hyperallfile, scoresfile, modelid,
              numaucthres, iternum, calib, params):
    if len(cvsets) == 0:
        print('No cross validation files')
        sys.exit()
    metrics = []
    cnt = 0
    print("Fit parameters : %s" % params)
    start_cv_all = time.time()
    prevcv=None
    for cvset in cvsets:
        print('%s Set: %s' % (runmode,cvset))
        if prevcv is None or prevcv['training']!=cvset['training']:
            prevcv=cvset
            traincolumns, model, metrics_dict_train, es_epochs = \
                   training(cvset, calc_test, modeltype, numaucthres, optimize_target, iternum, params)

        start_cv = time.time()
        tn = 0; fp = 0; fn = 0; tp = 0;
        y_scores = None; y_pred = None; y_val = None
        cvfiles = load_files(cvset, 'crossval', testsetdir)
        if len(cvfiles) == 0:
            print("No Validation dataset(s) found")
            return
        for cvfile in cvfiles:
            start_predict_file = time.time()
            print('Test/Validation File: %s' % cvfile)
            X_pd, y_pd, groups_pd = load_dataset(cvfile, params['feature_drop'], \
                                                 debug=debug, calib=calib)
            X_pd = X_pd.reindex(sorted(X_pd.columns), axis=1)
            valcolumns = X_pd.columns
            if debug:
                _fer=False
                for i in range(len(traincolumns)):
                    if traincolumns[i] != valcolumns[i]:
                        if not _fer:
                            print('Train - Test/Prediction columns mismatch')
                            print('Train Columns: %s' % traincolumns)
                            print('Test/Pred Columns: %s' % valcolumns)
                            _fer = True
                        print('WARNING! Training set column %d: %s is different from Validation Set Column %d: %s' % (
                        i, traincolumns[i], i, valcolumns[i]))

            X_val = X_pd.values
            _y_val = y_pd.values
            _y_val = _y_val[:, 0]
            X_pd = None; y_pd = None; groups_pd = None
            if debug:
                print("Running prediction...")
            _y_scores, _y_pred = run_predict(model, modeltype, X_val)
            #_y_scores, _y_pred = runmlprocess(run_predict_q,[model, modeltype, X_val],2)
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
            tsecs = time.time() - start_predict_file
            print("Predict time : %d min %d secs" % (int(tsecs/60.0), tsecs % 60))

        '''validation set metrics'''
        msetval = runmode
        metrics_dict_val = metrics_dict(*calc_metrics_custom(tn, fp, fn, tp, y_scores, y_val, \
                                                             numaucthres=numaucthres, debug=debug), msetval)
        metrics_dict_dist={}
        if numaucthres>0:
            metrics_dict_dist = metrics_dict_distrib(*calc_all_model_distrib(y_scores[:, 1], y_val, debug=debug), msetval)
        tsecs = time.time() - start_cv
        print("All files predict time : %d min %d secs" % (int(tsecs/60.0), tsecs % 60))
        print("Recall 1 val: %.3f, Recall 0 val: %.3f" % (metrics_dict_val['recall 1 %s'%msetval], metrics_dict_val['recall 0 %s'%msetval]))
        metrics_dict_fold = {'Model ID':'%d'%modelid, 'opt. metric': optimize_target}
        metrics_dict_fold['training set'] = '%s' % cvset['training']
        metrics_dict_fold[runmode+' set'] = '%s'%cvset['crossval']
        metrics_dict_fold = {**metrics_dict_fold, **metrics_dict_train, **metrics_dict_val, **metrics_dict_dist}
        if modeltype=='tf':
            metrics_dict_fold['early stop epochs'] = es_epochs
        metrics_dict_fold['params']='%s'%params
        metrics_dict_fold['CV Fit and predict min.']=(time.time() - start_cv)/60.0
        metrics.append(metrics_dict_fold)
        if writescores and numaucthres>0:
            sfile_suffix=''.join([ch for ch in '%s'%cvset['crossval'] if re.match(r'\w', ch)])
            cv_common.write_score(scoresfile+'_'+sfile_suffix+'.csv', None, None, y_val, y_scores[:,1])
    # final line in fold metrics
    metrics_dict_all = metrics_aggr2(metrics, msetval)
    metrics_dict_all[runmode + ' set'] = 'all set'
    metrics_dict_all['params'] = '%s'%params
    metrics_dict_all['Model ID'] = '%d'%modelid
    metrics_dict_all['opt. metric'] = optimize_target

    for k in metrics[0]:
        if not k in metrics_dict_all: metrics_dict_all[k]=''
    mean_metrics = {'Model ID': '%d'%modelid, 'opt. metric': optimize_target}
    mean_metrics = metrics_aggr(metrics, mean_metrics)
    mean_metrics["CV time (min)"] = (time.time() - start_cv_all) / 60.0
    mean_metrics['params'] = '%s'%params
    mean_metrics['Model ID'] = modelid
    print("All set's Recall 1 : %.3f, Recall 0 : %.3f" % \
          (metrics_dict_all['recall 1 %s' % msetval], metrics_dict_all['recall 0 %s' % msetval]))
    print('Mean %s : %s' % (optimize_target, mean_metrics[optimize_target]))
    metrics.append(metrics_dict_all) # append after mean metrics for writing
    cv_common.writemetrics(metrics, mean_metrics, hyperresfile, hyperallfile)

    return {
        'loss': -mean_metrics[optimize_target],
        'status': True,
        # -- store other results like this
        # 'eval_time': time.time(),
        'metrics': mean_metrics,
        'params': '%s' % params,
        'allmetrics': metrics,
        # -- attachments are handled differently
        # 'attachments':
        #    {'time_module': pickle.dumps(time.time)}
    }

testsets, space, testmodels, testfpattern, filters, nbest, changeparams, max_trials, calc_test, recmetrics, trainsetdir, testsetdir, \
numaucthres, modeltype, filedesc, runmode, writescores, resdir, iternum, calib, debug = space_new_test.create_space()

#tf.config.threading.set_inter_op_parallelism_threads(
#   8
#)
opt_targets = ['%s %s'%(ot,runmode) for ot in recmetrics]
if testfpattern is not None:
    testmodels = best_models.retrieve_best_models(resdir, testfpattern, recmetrics, 'val.', 'test', filters, nbest)
opt_targets = testmodels.keys()
hyperresfile = cv_common.get_filename(runmode, modeltype, filedesc, aggr='mean', resultsfolder=resdir)
hyperallfile = cv_common.get_filename(runmode, modeltype, filedesc, aggr='all', resultsfolder=resdir)
runtimes=1
for opt_target in opt_targets:
    scoresfile = cv_common.get_filename(opt_target, modeltype, filedesc, aggr='scores', ext='',
                                        resultsfolder=resdir)
    print("Output files : %s, %s"%(hyperresfile,hyperallfile))
    cnt=0
    for modelparams in testmodels[opt_target]:
        cnt+=1
        modelid = list(range(0,cnt)) if 'trial' not in modelparams.keys() else modelparams['trial']
        if changeparams is not None:
            for cp in changeparams: modelparams['params'][cp]=changeparams[cp]
        for i in range(runtimes):
            evalmodel(testsets, opt_target, calc_test, modeltype, hyperresfile, hyperallfile, \
                      scoresfile, modelid, numaucthres, iternum, calib, modelparams['params'])
