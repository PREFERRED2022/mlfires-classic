from hyperopt import hp

def create_space():

    newfeatures = ['xpos', 'ypos', 'month', 'wkd', 'lst', 'dew', 'freq', 'f81']
    categorical = ['dir_max', 'dom_dir','month', 'wkd', 'corine']
    problemfeat = ['dir_max', 'dom_dir','month', 'wkd']
    goodnewfeat = ['xpos', 'ypos','lst', 'dew', 'freq', 'f81']
    newfeatures = ['xpos', 'ypos', 'month', 'wkd', 'lst', 'dew', 'freq', 'f81']
    '''
    space = {'algo': hp.choice('algo', ['RF']),
             'n_estimators': hp.quniform('n_estimators', 50, 1050, 100),
             #'n_estimators': hp.choice('n_estimators', [50, 100, 120, 150, 170, 200, 250, 350, 500, 750, 1000, 1400, 1500]),
             'min_samples_split': hp.loguniform('min_samples_split',2, 2000),
             #'min_samples_split': hp.choice('min_samples_split',
             #                               [2, 10, 50, 70, 100, 120, 150, 180, 200, 250, 400, 600, 1000, 1300, 2000]),

             'min_samples_leaf': hp.loguniform('min_samples_leaf', 1, 2000),
             #'min_samples_leaf' :hp.choice('min_samples_leaf',[1, 10,30,40,50,100,120,150]),
             'criterion':hp.choice('criterion',["gini", "entropy"]),
             'max_features':hp.quniform('max_features', 1,10,1),
             'bootstrap':hp.choice('bootstrap',[True, False]),
             'max_depth': hp.choice('max_depth', [10, 20, 100, 200, 400, 500, 700, 1000, 1200, 2000, None]),
             'feature_drop': hp.choice('feature_drop', [[]]),
             'class_weight':hp.choice('class_weight',[{0:1,1:1}, {0:1,1:9},{0:1,1:300},{0:1,1:400},{0:1,1:500},{0:1,1:1000}])
             }

    space = {'algo': hp.choice('algo', ['RF']),
             'n_estimators': hp.choice('n_estimators', [120]),
             'min_samples_split': hp.choice('min_samples_split', [150]),
             'min_samples_leaf': hp.choice('min_samples_leaf', [30]),
             'criterion': hp.choice('criterion', ["entropy"]),
             'max_features': hp.choice('max_features', [1.5]),
             'bootstrap': hp.choice('bootstrap', [False]),
             'max_depth': hp.choice('max_depth', [1000]),
             'class_weight': hp.choice('class_weight', [{0: 2, 1: 8}]),
             'feature_drop': hp.choice('feature_drop', [[], newfeatures, problemfeat, ['wkd'], ['month'], ['dir_max'], ['dom_dir'], ['month', 'wkd'], ['dir'],\
                                                       list(set(newfeatures)-set(['dew'])), list(set(newfeatures)-set(['lst'])),
                                                       list(set(newfeatures)-set(['freq'])), list(set(newfeatures)-set(['f81'])),
                                                       list(set(newfeatures)-set(['xpos'])), list(set(newfeatures)-set(['ypos'])),
                                                       list(set(newfeatures)-set(['xpos','ypos'])), list(set(newfeatures)-set(['dew','lst'])),
                                                       list(set(newfeatures)-set(['freq','f81'])), problemfeat+['aspect']
                                                       ]),

             }
    '''
    space = { 'algo': hp.choice('algo', ['XT']),
        'n_estimators': hp.choice('n_estimators',[10, 20, 40, 60, 80, 100, 200, 400, 600, 800, 1000]),
        'criterion': hp.choice('criterion',['gini', 'entropy']),
        'max_depth': hp.quniform('max_depth',2, 40, 2),
        'min_samples_split': hp.choice('min_samples_split',[2, 10, 50, 70, 100, 120, 150, 180, 200, 250, 400, 600, 1000, 1300, 2000]),
        'min_samples_leaf': hp.choice('min_samples_leaf',[5, 10, 15, 20, 25, 30, 35, 40, 45]),
        'max_features': hp.quniform('max_features', 1,10,1),
        'bootstrap': hp.choice('bootstrap',[True, False]),
         #'oob_score': [True, False],
        'class_weights': hp.choice('class_weights',[{0: 4, 1: 6}, {0: 1, 1: 10}, {0: 1, 1: 50}, {0: 1, 1: 70}]),
        'feature_drop': [],
    }
    max_trials = 1
    testsets = {'balanced':'/work2/pa21/sgirtsou/production/datasets/randomnofire/old_random_new_feat_from_months.csv', 'imbalanced':'dataset_1_10_corine_level2_onehotenc.csv'}
    resultsdir = '/work2/pa21/sgirtsou/production/results/kfoldcv'
    #testsets = {'balanced':'/work2/pa21/sgirtsou/production/datasets/randomnofire/old_random_new_features_norm.csv', 'imbalanced':'dataset_1_10_corine_level2_onehotenc.csv'}
    calc_train_metrics = True
    #opt_targets = ['hybrid1 val', 'hybrid2 val', 'f1-score 1 val.', 'auc val.', 'recall 1 val.']
    opt_targets = ['auc val.']
    #modeltype = 'tensorflow'
    modeltype = 'sklearn'
    nfolds = 5
    desc = 'ETtest'
    writescores = True
    return 'balanced', testsets, nfolds, space, max_trials, calc_train_metrics, opt_targets, modeltype, desc, writescores, resultsdir
