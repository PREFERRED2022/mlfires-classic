#!/usr/bin/env python
from hyperopt import Trials, fmin, tpe, rand, STATUS_OK
from sklearn.model_selection import train_test_split, KFold, GroupKFold
import numpy as np
import time
import space_new as space
from functools import partial
from manage_model import create_model, run_predict_and_metrics, run_predict, fit_model, create_and_fit, allowgrowthgpus
import cv_common
from MLscores import metrics_aggr
from check_and_prepare_dataset import load_dataset
import gc
import tensorflow as tf
import multiprocessing as mp

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

def validatemodel(cv, X_pd, y_pd, groups_pd, optimize_target, calc_test, modeltype,
                  hpresfile, allresfile, scoresfile, trials, params):

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
    print("Fit parameters : %s" % params)

    if len(params['feature_drop'])>0:
        dropcols = [c for c in X_pd.columns if any([fd in c for fd in params['feature_drop']])]
        print("dropping columns: %s"%dropcols)
        X_pd = X_pd.drop(columns=dropcols)

    X = X_pd.values
    y = y_pd.values
    groups = groups_pd.values
    Xhash = cv_common.gethashdict(X)
    y_scores_all = np.zeros(y.shape[0])

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

        '''       
        mp.set_start_method('spawn')
        q=mp.Queue()
        fitproc=mp.Process(target=create_and_fit,args=(modeltype, params, X_train, y_train, X_val, y_val, q))

        fitproc.start()
        fitproc.join()
        model=q.get()
        res=q.get()
        '''

        if modeltype == 'tf':
            es_epochs = len(res.history['loss'])

        print("Fit time (min): %.1f"%((time.time() - start_time)/60.0))
        start_time = time.time()

        '''training set metrics'''
        metrics_dict_train, y_scores = run_predict_and_metrics(model, modeltype, X_train, y_train, 'train', not calc_test)
        print("Training metrics time (min): %.1f"%((time.time() - start_time)/60.0))
        start_time = time.time()

        '''validation set metrics'''
        valset = 'val.'
        metrics_dict_val, y_scores = run_predict_and_metrics(model, modeltype, X_val, y_val, valset)
        cv_common.updateYrows(X_val, y_scores[:, 1], Xhash, y_scores_all)

        print("Validation metrics time (min): %.1f"%((time.time() - start_time)/60.0))
        print("Recall 1 val: %.3f, Recall 0 val: %.3f" % (metrics_dict_val['recall 1 %s'%valset],metrics_dict_val['recall 0 %s'%valset]))

        metrics_dict_fold = {'trial': '%d'%len(trials), 'opt. metric': optimize_target}
        metrics_dict_fold['fold'] = 'fold %d'%cnt
        metrics_dict_fold = {**metrics_dict_fold, **metrics_dict_train, **metrics_dict_val}
        if modeltype=='tf':
            metrics_dict_fold['early stop epochs'] = es_epochs
        metrics_dict_fold['params']='%s'%params
        metrics_dict_fold['CV fold fit time']=(time.time() - start_fold_time)/60.0
        metrics.append(metrics_dict_fold)

    mean_metrics = {'trial': '%d'%len(trials), 'opt. metric': optimize_target}
    mean_metrics = metrics_aggr(metrics, mean_metrics, hybrid_on_aggr=True, y_scores=y_scores_all, y=np.transpose(y)[0], valst=valset)
    mean_metrics["CV time (min)"] = (time.time() - start_folds)/60.0
    mean_metrics['params'] = '%s' % params

    print('Mean %s : %.4f' % (optimize_target,mean_metrics[optimize_target]))
    cv_common.writemetrics(metrics, mean_metrics, hpresfile, allresfile)
    if writescores:
        cv_common.write_score(scoresfile, groups_pd, y_pd, y_scores_all)
    gc.collect()
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
iteration = 0

#load hyperparaeters
tset, testsets, num_folds, space, max_trials, hypalgoparam, calc_test, opt_targets, modeltype, desc, \
writescores, resultsfolder, GPUMBs = space.create_space()

#initialize cross validation folds
kf = GroupKFold(n_splits=num_folds)

# load training/validation dataset
dsfile = testsets[tset]
X_pd, y_pd, groups_pd = load_dataset(dsfile, featuredrop=[], class0nrows=0, debug=True)
pdscores=None

# run tests per optimization metrics
runmode ='val.'
opt_targets = ['%s %s'%(ot,runmode) for ot in opt_targets]
allowgrowthgpus()
for opt_target in opt_targets:
    hpresfile = cv_common.get_filename(opt_target, modeltype, desc, aggr='mean', resultsfolder=resultsfolder)
    allresfile = cv_common.get_filename(opt_target, modeltype, desc, aggr='all', resultsfolder=resultsfolder)
    scoreresfile = cv_common.get_filename(opt_target, modeltype, desc, aggr='scores', resultsfolder=resultsfolder)
    trials = Trials()

    # prepear validate function
    validatemodelpart = partial(validatemodel, kf, X_pd, y_pd, groups_pd, opt_target, calc_test, modeltype,
                                hpresfile, allresfile, scoreresfile, trials)

    hypalgo = cv_common.get_hyperopt_algo(hypalgoparam)
    if hypalgo is None:
        print('Wrong optimization algorithm')
        break

    fmin(fn=validatemodelpart,  # function to optimize
        space=space,
        algo=hypalgo,  # optimization algorithm, hyperotp will select its parameters automatically
        max_evals=max_trials,  # maximum number of iterations
        trials=trials,  # logging
        #rstate=np.random.RandomState(random_state)  # fixing random state for the reproducibility
        #rstate = np.random.default_rng(seed=random_state)
        )

