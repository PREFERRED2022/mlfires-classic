from hyperopt import hp

def create_space():
    '''
    space = {'algo': hp.choice('algo', ['RF']),
             'n_estimators': hp.choice('n_estimators', [50, 100, 120, 150,170,200, 250, 350, 500, 750, 1000,1400, 1500]),
              'min_samples_split': hp.choice('min_samples_split',[2, 10, 50, 70,100,120,150,180, 200, 250,400,600,1000, 1300, 2000]),
              'min_samples_leaf' :hp.choice('min_samples_leaf',[1, 10,30,40,50,100,120,150]),
              'criterion':hp.choice('criterion',["gini", "entropy"]),
              'max_features':hp.quniform('max_features', 1,10,1),
              'bootstrap':hp.choice('bootstrap',[True, False]),
              'max_depth': hp.choice('max_depth', [10, 20, 100, 200, 400,500, 700, 1000, 1200,2000, None]),
              'feature_drop': hp.choice('feature_drop', [[]]),
              'class_weights':hp.choice('class_weight',[{0:1,1:300},{0:1,1:400},{0:1,1:500},{0:1,1:1000}])
            }
    
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
    '''
    space = { 'algo': hp.choice('algo', ['XGB']),
        'max_depth': hp.quniform('max_depth',2, 100, 2),
        'n_estimators': hp.choice('n_estimators',[10, 20, 40, 60, 80, 100, 200, 400, 600, 800, 1000]),
        #'scale_pos_weight': range(1, 400, 50),
        'subsample': hp.choice('subsample',[0.5, 0.6, 0.7, 0.8, 0.9, 1]),
        'alpha': hp.choice('alpha', [0, 1, 10, 20, 40, 60, 80, 100]),
        'gamma': hp.choice('gamma',[0, 0.001, 0.01, 0.1, 1, 10, 100, 1000]),
        'lambda': hp.quniform('lambda',1, 22, 1),
        #'scale_pos_weight': [6,7,8,9,15,50,70]
        'scale_pos_weight': hp.choice('scale_pos_weight',[9, 15, 50, 70, 100, 200, 500]),
        'feature_drop': [],
        }
    
    max_trials = 300
    #trainsetdir = '/work2/pa21/sgirtsou/production/datasets/hard_cosine_similarity'
    trainsetdir = '/work2/pa21/sgirtsou/production/datasets/randomnofire'
    testsetdir = '/work2/pa21/sgirtsou/production'
    testsets = [#{'training':['old_dataset_2010.csv','old_dataset_2011.csv','old_dataset_2012.csv'],\
                # 'crossval':['*august_2013_norm.csv']},
                #{'training':['old_dataset_2010.csv','old_dataset_2011.csv','old_dataset_2012.csv','old_dataset_2013.csv'],\
                # 'crossval':['*august_2014_norm.csv']},
                {'training':['old_dataset_2010.csv','old_dataset_2011.csv','old_dataset_2012.csv','old_dataset_2013.csv','old_dataset_2014.csv'],\
                 'crossval':['*august_2015_norm.csv']},
                {'training':['old_dataset_2010.csv','old_dataset_2011.csv','old_dataset_2012.csv','old_dataset_2013.csv','old_dataset_2014.csv','old_dataset_2015.csv'],\
                 'crossval':['*august_2016_norm.csv']},
                {'training':['old_dataset_2010.csv','old_dataset_2011.csv','old_dataset_2012.csv','old_dataset_2013.csv','old_dataset_2014.csv','old_dataset_2015.csv','old_dataset_2016.csv'],\
                 'crossval':['*august_2017_norm.csv']},
                {'training':['old_dataset_2010.csv','old_dataset_2011.csv','old_dataset_2012.csv','old_dataset_2013.csv','old_dataset_2014.csv','old_dataset_2015.csv','old_dataset_2016.csv','old_dataset_2017.csv'],\
                 'crossval':['*august_2018_norm.csv']}
               ]
    '''
    testsets = [{'training':['*2010_norm.csv','*2011_norm.csv','*2012_norm.csv'],\
                 'crossval':['*august_2013_norm.csv']},
                {'training':['*2010_norm.csv','*2011_norm.csv','*2012_norm.csv','*2013_norm.csv'],\
                 'crossval':['*august_2014_norm.csv']},
                {'training':['*2010_norm.csv','*2011_norm.csv','*2012_norm.csv','*2013_norm.csv','*2014_norm.csv'],\
                 'crossval':['*august_2015_norm.csv']},
                {'training':['*2010_norm.csv','*2011_norm.csv','*2012_norm.csv','*2013_norm.csv','*2014_norm.csv','*2015_norm.csv'],\
                 'crossval':['*august_2016_norm.csv']},
                {'training':['*2010_norm.csv','*2011_norm.csv','*2012_norm.csv','*2013_norm.csv','*2014_norm.csv','*2015_norm.csv','*2016_norm.csv'],\
                 'crossval':['*august_2017_norm.csv']},
                {'training':['*2010_norm.csv','*2011_norm.csv','*2012_norm.csv','*2013_norm.csv','*2014_norm.csv','*2015_norm.csv','*2016_norm.csv','*2017_norm.csv'],\
                 'crossval':['*august_2018_norm.csv']}
               ]
    '''
    calc_train_metrics = True
    runmode = 'val.'
    #opt_targets = ['hybrid2', 'hybrid5', 'NH2', 'NH5', 'NH10', 'auc']
    opt_targets = ['hybrid2', 'hybrid5', 'NH2']
    #opt_targets = ['NH5', 'NH10', 'auc']
    debug = True
    modeltype = 'sk'
    #modeltype = 'tf'
    class0_headrows = 1000000
    filespec = "XT_1M_nofeatdrop"
    aucthress = 30
    writescore = False
    resdir = '/work2/pa21/sgirtsou/production/results/newcv'
    suggestalgo = 'tpe'
    return testsets, space, None, None, [], max_trials, suggestalgo, calc_train_metrics, opt_targets, trainsetdir, testsetdir,\
           aucthress, modeltype, class0_headrows, filespec, runmode, writescore, resdir, debug


