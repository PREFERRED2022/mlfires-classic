from hyperopt import hp,tpe

def create_space():
    problemfeatures = ['month', 'weekday', 'dom_dir', 'dir_max']
    dropfeat1 = ['month', 'weekday', 'dom_dir', 'dir_max']+['corine_%d' % i for i in range(1, 10)]
    dropfeat2 = ['month', 'weekday', 'dom_dir', 'dir_max']+['corine_gr%d' % i for i in range(1, 10)]
    dropfeat3 = ['month', 'weekday', 'dom_dir', 'dir_max', 'pop']+['corine_%d' % i for i in range(1, 10)]
    dropfeat4 = ['month', 'weekday', 'dom_dir', 'dir_max', 'pop']+['corine_gr%d' % i for i in range(1, 10)]

    '''
    space = {'algo': hp.choice('algo', ['RF']),
             'n_estimators': hp.choice('n_estimators', [50, 100, 120, 150,170,200, 250, 350, 500, 1000]),
              'min_samples_split': hp.choice('min_samples_split',[2, 10, 50, 70,100,120,150,180, 200, 250,400,600,1000, 2000]),
              'min_samples_leaf' :hp.choice('min_samples_leaf',[1, 10,30,40,50,100,120,150]),
              'criterion':hp.choice('criterion',["gini", "entropy"]),
              'max_features':hp.quniform('max_features', 1,10,1),
              'bootstrap':hp.choice('bootstrap',[True, False]),
              'max_depth': hp.choice('max_depth', [10, 20, 100, 200, 400,500, 700, 1000, None]),
              'feature_drop': hp.choice('feature_drop',[dropfeat1,dropfeat2,dropfeat3,dropfeat4]),
              'class_weights':hp.choice('class_weight',[{0:1,1:1}, {0:1,1:10}, {0:1,1:100},{0:1,1:500}, {0:1,1:1000}])
            }
    
    '''
    space = { 'algo': hp.choice('algo', ['XT']),
        'n_estimators': hp.choice('n_estimators',[10, 20, 40, 60, 80, 100, 200, 400, 800]),
        'criterion': hp.choice('criterion',['gini', 'entropy']),
        'max_depth': hp.choice('max_depth',[None, 2, 10, 20, 50, 100, 400]),
        'min_samples_split': hp.choice('min_samples_split',[2, 10, 50, 70, 100, 120, 150, 180, 200, 250, 400, 600, 1000, 1300, 2000]),
        'min_samples_leaf': hp.choice('min_samples_leaf',[1, 10,30,40,50,100,120,150]),
        'max_features': hp.quniform('max_features', 1,10,1),
        'bootstrap': hp.choice('bootstrap',[True, False]),
         #'oob_score': [True, False],
        'class_weights': hp.choice('class_weights',[{0: 1, 1: 1}, {0: 1, 1: 10}, {0: 1, 1: 50}, {0: 1, 1: 70}]),
        'feature_drop': hp.choice('feature_drop',[dropfeat1,dropfeat2,dropfeat3,dropfeat4]),
     }
    '''
    space = {'algo': hp.choice('algo', ['XGB']),
             'max_depth': hp.quniform('max_depth', 2, 100, 2),
             #'n_estimators': hp.choice('n_estimators', [10, 20, 40, 60, 80, 100, 200, 400, 600, 800, 1000]),
             'n_estimators': hp.choice('n_estimators', [10, 20, 40, 80, 160, 320, 640]),
             # 'scale_pos_weight': range(1, 400, 50),
             'subsample': hp.choice('subsample', [0.5, 0.6, 0.7, 0.8, 0.9, 1]),
             'alpha': hp.choice('alpha', [0, 1, 10, 20, 40, 60, 80, 100]),
             'gamma': hp.choice('gamma', [0, 0.001, 0.01, 0.1, 1, 10, 100, 1000]),
             'lambda': hp.quniform('lambda', 1, 22, 1),
             # 'scale_pos_weight': [6,7,8,9,15,50,70]
             'scale_pos_weight': hp.choice('scale_pos_weight', [9, 15, 50, 70, 100, 200, 500]),
             'feature_drop': hp.choice('feature_drop',[dropfeat1,dropfeat2,dropfeat3,dropfeat4]),
             }

    '''
    max_trials = 500
    #testsets = {'balanced':'/home/aapos/Documents/newcrossval/datasets/randomnofire/old_random_new_features_norm.csv', 'imbalanced':'dataset_1_10_corine_level2_onehotenc.csv'}
    testsets = {'balanced':'/mnt/nvme2tb/ffp/datasets/train/train_new_sample_1_2_norm.csv', 'imbalanced':'dataset_1_10_corine_level2_onehotenc.csv'}
    calc_train_metrics = True
    #opt_targets = ['hybrid2 val', 'hybrid5 val', 'f1-score 1 val.', 'auc val.', 'recall 1 val.']
    #opt_targets = ['auc val.']
    opt_targets = ['hybrid1','hybrid2', 'hybrid5', 'NH2', 'NH5','NH10', 'auc', 'f1-score 1']
    #opt_targets = ['hybrid1']
    modeltype = 'sk'
    description = 'XT_ns'
    nfolds = 5
    writescores = False
    resultsfolder = '/mnt/nvme2tb/ffp/results/newdefCV'
    #hypalgo = 'random'
    hypalgo = 'tpe'
    gpuMBs = 10
    return 'balanced', testsets, nfolds, space, max_trials, hypalgo, calc_train_metrics, opt_targets, modeltype, description,\
           writescores, resultsfolder, gpuMBs

