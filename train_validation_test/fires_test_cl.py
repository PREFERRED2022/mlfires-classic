#from hyperopt import Trials, fmin, tpe, STATUS_OK
import numpy as np
import os
import time
import cl_space_new_test
import re
from manage_model import run_predict_q, run_predict_and_metrics_q, create_and_fit_q, run_predict, \
    run_predict_and_metrics, fit_model, create_model, allowgrowthgpus, mm_save_model, mm_save_weights, mm_load_model
import fileutils
from MLscores import calc_metrics_custom, cmvals, metrics_aggr, metrics_aggr2, \
    metrics_dict, calc_all_model_distrib, metrics_dict_distrib
import sys
from check_and_prepare_dataset import load_dataset
import cv_common
import best_models
import multiprocessing as mp
import pandas as pd
from fileutils import del_file_or_folder

def load_files(cvset, settype, setdir):
    setfiles = []
    for dsfilepattern in cvset[settype]:
        setfiles += [f for f in fileutils.find_files(setdir, dsfilepattern, listtype="walk")]
    return sorted(setfiles)

def resfilename(opt_target, filedesc):
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

def training(env, cvset, optimize_target, params, modelfile):
    trfiles = load_files(cvset, 'training', env.trainsetdir)
    if len(trfiles) == 0:
        print("No training dataset(s) found")
        return
    print('Training Files: %s' % trfiles)
    bestmodel = None
    prevmax = None

    allowgrowthgpus()
    for i in range(env.iternum):
        X_pd, y_pd, groups_pd = load_dataset(trfiles, params['feature_drop'], debug=env.debug)
        X_pd = X_pd.reindex(sorted(X_pd.columns), axis=1)
        traincolumns = X_pd.columns

        X_train = X_pd.values
        y_train = y_pd.values
        y_train = y_train[:, 0]
        start_fit = time.time()
        print("Fitting model ...")

        model = None
        model = create_model(env.modeltype, params, X_train)
        model, res = fit_model(env.modeltype, model, params, X_train, y_train, X_train, y_train)

        # res = runmlprocess(create_and_fit_q, [modeltype, params, X_train, y_train, X_train, y_train], 1)

        tsecs = time.time() - start_fit
        print("Fit time : %d min %d secs" % (int(tsecs/60.0), tsecs % 60))

        '''training set metrics'''
        start_time_trmetrics = time.time()
        mset = 'train'
        metrics_dict_train, y_scores = run_predict_and_metrics(model, env.modeltype, X_train, y_train, 'train', not env.calc_test)

        # metrics_dict_train, y_scores = runmlprocess(run_predict_and_metrics_q, [modeltype, X_train, y_train, 'train',
        #                                                       not calc_test, numaucthres],2)
        tsecs = time.time() - start_time_trmetrics
        print("Training metrics time : %d min %d secs" % (int(tsecs/60.0), tsecs % 60))

        if env.debug:
            calc_metrics_custom(metrics_dict_train['TN %s' % mset], metrics_dict_train['FP %s' % mset], \
                                metrics_dict_train['FN %s' % mset], metrics_dict_train['TP %s' % mset], y_scores, y_train, \
                                numaucthres=env.numaucthres, debug=env.debug)
        if env.modeltype == 'tf':
            es_epochs = len(res.history['loss'])
        else:
            es_epochs = 0

        all_metrics = metrics_aggr([metrics_dict_train], {})
        trtarget = optimize_target.replace('test','train')
        if prevmax is None or all_metrics[trtarget] > prevmax:
            bestmodel = model
            del_file_or_folder(modelfile)
            mm_save_model(modelfile, bestmodel, env.modeltype, params)
            bestmetrics = metrics_dict_train
            bestepochs = es_epochs
            bestiter=i
            prevmax = all_metrics[trtarget]
        print("Training Iteration #%d/%d. BEST iteration so far (#%d) for %s : %.3f" %
              (i+1, env.iternum, bestiter+1, optimize_target, prevmax))
    return traincolumns, bestmodel, bestmetrics, bestepochs


def evalmodel(env, optimize_target, hyperresfile, hyperallfile, scoresfile, modelfile,
              modelid, params):

    if len(env.testsets) == 0:
        print('No cross validation files')
        sys.exit()
    metrics = []
    cnt = 0
    print("Fit parameters : %s" % params)
    start_cv_all = time.time()
    prevcv=None
    for tset in env.testsets:
        print('%s Set: %s' % (env.runmode,tset))
        if (prevcv is None or prevcv['training']!=tset['training']) and not os.path.exists(modelfile):
            prevcv=tset
            traincolumns, model, metrics_dict_train, es_epochs = \
                   training(env, tset, optimize_target, params, modelfile)
        # load already saved model
        if os.path.exists(modelfile):
            model=mm_load_model(modelfile, env.modeltype, params)
            metrics_dict_train={}
            traincolumns=None
            es_epochs=0

        start_cv = time.time()
        tn = 0; fp = 0; fn = 0; tp = 0;
        y_scores = None; y_pred = None; y_val = None
        cvfiles = load_files(tset, 'crossval', env.testsetdir)
        if len(cvfiles) == 0:
            print("No Validation dataset(s) found")
            return
        for cvfile in cvfiles:
            start_predict_file = time.time()
            print('Test/Validation File: %s' % cvfile)

            X_pd, y_pd, groups_pd, pdid = load_dataset(cvfile, params['feature_drop'], \
                                                 debug=env.debug, returnid=True, calib=env.calib)
            X_pd = X_pd.reindex(sorted(X_pd.columns), axis=1)
            valcolumns = X_pd.columns
            if env.debug:
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
            if env.writescore:
                _groups_pd = groups_pd
                _pdid = pdid
            X_pd = None; y_pd = None; groups_pd = None; pdid=None
            if env.debug:
                print("Running prediction...")
            _y_scores, _y_pred = run_predict(model, env.modeltype, X_val)
            #_y_scores, _y_pred = runmlprocess(run_predict_q,[model, modeltype, X_val],2)
            if env.numaucthres>0:
                if y_scores is None:
                    y_scores = _y_scores
                    y_pred = _y_pred
                    y_val = _y_val
                    if env.writescore:
                        all_pdid = _pdid
                        all_groups_pd = _groups_pd
                else:
                    y_scores = np.concatenate((y_scores, _y_scores))
                    y_pred = np.concatenate((y_pred, _y_pred))
                    y_val = np.concatenate((y_val, _y_val))
                    if env.writescore:
                        all_groups_pd = pd.concat([all_groups_pd, _groups_pd])
                        all_groups_pd=all_groups_pd.reset_index(drop=True)
                        all_pdid = pd.concat([all_pdid, _pdid])
                        all_pdid=all_pdid.reset_index(drop=True)
            if env.debug:
                print("confusion matrix retrieval...")
            _tn, _fp, _fn, _tp = cmvals(_y_val, _y_pred)
            if env.debug:
                print("file tn : %d, fp : %d, fn : %d, tp : %d" % (_tn, _fp, _fn, _tp))
            tn += _tn;
            fp += _fp;
            fn += _fn;
            tp += _tp;
            if env.debug:
                print("sums tn : %d, fp : %d, fn : %d, tp : %d" % (tn, fp, fn, tp))
            tsecs = time.time() - start_predict_file
            print("Predict time : %d min %d secs" % (int(tsecs/60.0), tsecs % 60))

        '''validation set metrics'''
        msetval = env.runmode
        metrics_dict_val = metrics_dict(*calc_metrics_custom(tn, fp, fn, tp, y_scores, y_val, \
                                                             numaucthres=env.numaucthres, debug=env.debug), msetval)
        metrics_dict_dist={}
        if env.numaucthres>0:
            metrics_dict_dist = metrics_dict_distrib(*calc_all_model_distrib(y_scores[:, 1], y_val, debug=env.debug), msetval)
        tsecs = time.time() - start_cv
        print("All files predict time : %d min %d secs" % (int(tsecs/60.0), tsecs % 60))
        print("Recall 1 val: %.3f, Recall 0 val: %.3f" % (metrics_dict_val['recall 1 %s'%msetval], metrics_dict_val['recall 0 %s'%msetval]))
        metrics_dict_fold = {'Model ID':'%s'%modelid, 'opt. metric': optimize_target}
        metrics_dict_fold['training set'] = '%s' % tset['training']
        metrics_dict_fold[env.runmode+' set'] = '%s'%tset['crossval']
        metrics_dict_fold = {**metrics_dict_fold, **metrics_dict_train, **metrics_dict_val, **metrics_dict_dist}
        if env.modeltype=='tf':
            metrics_dict_fold['early stop epochs'] = es_epochs
        metrics_dict_fold['params']='%s'%params
        metrics_dict_fold['CV Fit and predict min.']=(time.time() - start_cv)/60.0
        metrics.append(metrics_dict_fold)
        if env.writescore and env.numaucthres>0:
            sfile_suffix=''.join([ch for ch in '%s'%tset['crossval'] if re.match(r'\w', ch)])
            cv_common.write_score(scoresfile+'_'+sfile_suffix+'.csv', all_groups_pd,
                                  y_val, y_scores[:,1], all_pdid, modelid)
            print("Write scores to %s" % scoresfile+'_'+sfile_suffix+'.csv')
    # final line in fold metrics
    metrics_dict_all = metrics_aggr2(metrics, msetval)
    metrics_dict_all[env.runmode + ' set'] = 'all set'
    metrics_dict_all['params'] = '%s'%params
    metrics_dict_all['Model ID'] = '%s'%modelid
    metrics_dict_all['opt. metric'] = optimize_target

    for k in metrics[0]:
        if not k in metrics_dict_all: metrics_dict_all[k]=''
    mean_metrics = {'Model ID': '%s'%modelid, 'opt. metric': optimize_target}
    mean_metrics = metrics_aggr(metrics, mean_metrics)
    mean_metrics["CV time (min)"] = (time.time() - start_cv_all) / 60.0
    mean_metrics['params'] = '%s'%params
    mean_metrics['Model ID'] = modelid
    print("All set's Recall 1 : %.3f, Recall 0 : %.3f" % \
          (metrics_dict_all['recall 1 %s' % msetval], metrics_dict_all['recall 0 %s' % msetval]))
    print('Mean %s : %s' % (optimize_target, mean_metrics[optimize_target]))
    metrics.append(metrics_dict_all) # append after mean metrics for writing
    print("write mean metrics to %s"%hyperresfile)
    print("and all datasets to %s"% hyperallfile)
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

def get_modelext(modeltype, params, modelid, runid):
   if modeltype == 'tf':
        ext='.h5'
        ftype = 'model_' + 'id_' + str(modelid) + '_r_' + str(runid)
   elif modeltype == 'tfw':
        ext='.cpkt'
        ftype='weights_'+'id_'+str(modelid)+'_r_'+str(runid)
   elif modeltype == 'sk':
      ftype = params['algo'] + 'model_' + 'id_' + str(modelid) + '_r_' + str(runid)
      if params['algo']!='XGB':
          ext = '.pickle'
      else:
          ext = '.json'
   return ftype, ext



def main(args):

    # override configuration with arguments
    recmetrics = None
    algo = None
    writescore = None
    region = None

    if len(args) >= 1:
        recmetrics = eval(args[0])
    if len(args) >= 2:
        algo = args[1]
    if len(args) >= 3:
        writescore = eval(args[2])
    if len(args) >= 4:
        region = args[3]

    # space configuration
    env = cl_space_new_test.space(recmetrics=recmetrics, algo=algo, writescore=writescore, region=region)

    if env.testfpattern is not None:
        testmodels = best_models.retrieve_best_models(env.resdir, env.testfpattern, env.recmetrics, 'val.', 'test', env.filters, env.nbest)
    opt_targets = testmodels.keys()
    hyperresfile = cv_common.get_filename(env.runmode, env.modeltype, env.filespec, ftype='mean', folder=os.path.join(env.resdir,env.runmode))
    hyperallfile = cv_common.get_filename(env.runmode, env.modeltype, env.filespec, ftype='all', folder=os.path.join(env.resdir,env.runmode))
    modelfolder = os.path.join(os.path.dirname(os.path.dirname(hyperresfile)), 'entiremodels')
    weightsfolder = os.path.join(os.path.dirname(os.path.dirname(hyperresfile)), 'weights')
    if not os.path.isdir(modelfolder): os.makedirs(modelfolder)
    if not os.path.isdir(weightsfolder): os.makedirs(weightsfolder)
    runtimes=1
    for opt_target in opt_targets:
        scoresfile = cv_common.get_filename(opt_target, env.modeltype, env.filespec, ftype='scores', ext='',
                                            folder=os.path.join(env.resdir,env.runmode))
        print("Output files : %s, %s"%(hyperresfile,hyperallfile))
        cnt=0
        for modelparams in testmodels[opt_target]:
            cnt+=1
            modelid = cnt if 'trial' not in modelparams.keys() else modelparams['trial']
            if env.changeparams is not None:
                for cp in env.changeparams: modelparams['params'][cp]=env.changeparams[cp]
            for i in range(runtimes):
                if env.modelfile is None:
                    ftype, ext = get_modelext(env.modeltype, modelparams['params'], modelid, i)
                    modelfile = cv_common.get_filename(opt_target, env.modeltype, env.filespec, ftype=ftype, ext=ext, folder=modelfolder)
                else: modelfile=env.modelfile
                modelidfull = str(modelid) + '_'+ str(i)
                evalmodel(env, opt_target, hyperresfile, hyperallfile, scoresfile, modelfile,\
                          modelidfull,  modelparams['params'])


if __name__ == '__main__':
    main(sys.argv[1:])