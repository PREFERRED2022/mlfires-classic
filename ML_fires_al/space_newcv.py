from hyperopt import hp

def create_space():

    space = {'n_internal_layers': hp.choice('n_internal_layers',
                [
                    #(0, {'layer_1_0_nodes': hp.quniform('layer_1_0_nodes', 10, 310, 50)}),
                    #(1, {'layer_1_1_nodes': hp.quniform('layer_1_1_nodes', 10, 310, 50), 'layer_2_1_nodes': hp.quniform('layer_2_1_nodes', 10, 310, 50)}),
                    #(2, {'layer_1_2_nodes': hp.quniform('layer_1_2_nodes', 10, 310, 50), 'layer_2_2_nodes': hp.quniform('layer_2_2_nodes', 10, 310, 50),
                    #     'layer_3_2_nodes': hp.quniform('layer_3_2_nodes', 10, 310, 50)})

                    (0, {'layer_1_0_nodes': hp.choice('layer_1_0_nodes', [50] )}),
                    #(1, {'layer_1_1_nodes': hp.choice('layer_1_1_nodes', [500, 1000]),
                    #     'layer_2_1_nodes': hp.choice('layer_2_1_nodes', [500, 1000])}),
                    #(2, {'layer_1_2_nodes': hp.choice('layer_1_2_nodes', [500, 1000]),
                    #     'layer_2_2_nodes': hp.choice('layer_2_2_nodes', [500, 1000]),
                    #     'layer_3_2_nodes': hp.choice('layer_3_2_nodes', [500, 1000])}),
                ]
                ),
             'dropout': hp.choice('dropout',[None]),
             #'class_weights': hp.choice('class_weights', [[1, 5],[1, 10], [1, 50], [1, 1]])
             #'class_weights': hp.choice('class_weights', [{0:1, 1:5}, {0:1, 1:10}, {0:1, 1:50}, {0:1, 1:1}]),
             'class_weights': hp.choice('class_weights', [{0:1, 1:1}, {0:2,1:3}, {0:3,1:7}, {0:1,1:4}, {0:1,1:9}, {0:1, 1:25}, {0:1, 1:50}, {0:1, 1:100} , {0:1, 1:200}]),
             #'feature_drop': hp.choice('feature_drop',['','bin','DIR','COR']),
             #'feature_drop': hp.choice('feature_drop', [[],['_dir_'],['aspect'], ['aspect', '_dir_']]),
             'feature_drop': hp.choice('feature_drop', [['wkd', 'month','f81','frequency','x','y']]),
             'max_epochs': hp.choice('max_epochs', [20]),
             #'metric': hp.choice('metric',['accuracy', 'sparse'])
             #'metric': hp.choice('metric', ['tn'])
             'metric': hp.choice('metric',['accuracy']),
             'optimizer': hp.choice('optimizer', [{'name': 'Adam',
                                                   'adam_params': hp.choice('adam_params',
                                                                            [None, {'learning_rate_adam': hp.uniform(
                                                                                'learning_rate_adam', 0.0001, 1), \
                                                                                    'beta_1': hp.uniform('beta_1',
                                                                                                         0.0001, 1),
                                                                                    'beta_2': hp.uniform('beta_2',
                                                                                                         0.0001, 1), \
                                                                                    'amsgrad': hp.choice('amsgrad',
                                                                                                         [True,
                                                                                                          False])}])},
                                                  {'name': 'SGD',
                                                   'learning_rate_SGD': hp.uniform('learning_rate_SGD', 0.0001, 1)}]),
             'ES_monitor': hp.choice('ES_monitor', ['loss']),  # 'val_loss','loss'
             'ES_patience': hp.choice('ES_patience', [10]),
             'ES_mindelta': hp.choice('ES_mindelta', [0.002]),
             'batch_size': hp.choice('batch_size', [512])
             #'feature_drop': hp.choice('feature_drop', ['bin'])
             }
    '''
    space = {'algo': hp.choice('algo', ['RF']),
             'n_estimators': hp.choice('n_estimators', [50, 100, 120, 150,170,200, 250, 350, 500, 750, 1000,1400, 1500]),
              'min_samples_split': hp.choice('min_samples_split',[2, 10, 50, 70,100,120,150,180, 200, 250,400,600,1000, 1300, 2000]),
              'min_samples_leaf' :hp.choice('min_samples_leaf',[1, 10,30,40,50,100,120,150]),
              'criterion':hp.choice('criterion',["gini", "entropy"]),
              'max_features':hp.quniform('max_features', 1,10,1),
              'bootstrap':hp.choice('bootstrap',[True, False]),
              'max_depth': hp.choice('max_depth', [10, 20, 100, 200, 400,500, 700, 1000, 1200,2000, None]),
              'feature_drop': hp.choice('feature_drop', [['wkd', 'month']]),
             'class_weight':hp.choice('class_weight',[{0:1,1:9},{0:1,1:300},{0:1,1:400},{0:1,1:500},{0:1,1:1000}])
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
        'feature_drop': ['wkd', 'month','f81','frequency','x','y'],
    }
    
    space = { 'algo': hp.choice('algo', ['XGB']),
        'max_depth': hp.quniform('max_depth',2, 100, 2),
        'n_estimators': hp.choice('n_estimators',[10, 20, 40, 60, 80, 100, 200, 400, 600, 800, 1000]),
        # 'scale_pos_weight': range(1, 400, 50),
        'subsample': hp.choice('subsample',[0.5, 0.6, 0.7, 0.8, 0.9, 1]),
        'alpha': hp.choice('alpha', [0, 1, 10, 20, 40, 60, 80, 100]),
        'gamma': hp.choice('gamma',[0, 0.001, 0.01, 0.1, 1, 10, 100, 1000]),
        'lambda': hp.quniform('lambda',1, 22, 1),
        # 'scale_pos_weight': [6,7,8,9,15,50,70]
        'scale_pos_weight': hp.choice('scale_pos_weight',[9, 15, 50, 70, 100, 200, 500]),
        'feature_drop': ['wkd', 'month','f81','frequency','x','y'],
        }
    '''
    #runmode = 'val.'
    runmode = 'test'
    testspace = {'hybrid2 %s'%runmode:
                 [{'n_internal_layers': (0, {'layer_1_0_nodes': 200}),'dropout': False,'class_weights': {0: 1, 1: 5},
                 'feature_drop': ['month', 'wkd','dir', 'pos', 'f81', 'frequency'],'metric': 'accuracy',
                 'optimizer': {'name': 'Adam','adam_params':None},
                 'max_epochs': 20,'ES_monitor':'loss','ES_patience':10,'ES_mindelta':0.002,'batch_size':512
                  }],
                 'auc %s'%runmode:
                  [{'n_internal_layers': (1, {'layer_1_1_nodes': 100, 'layer_2_1_nodes': 100}),'dropout': False,
                    'class_weights': {0: 1, 1: 5}, 'feature_drop': ['month', 'wkd','dir', 'pos', 'f81', 'frequency'],
                    'metric': 'accuracy','optimizer': {'name': 'Adam','adam_params':None},
                   'max_epochs': 20,'ES_monitor':'loss','ES_patience':10,'ES_mindelta':0.002,'batch_size':512
                  }]}


    max_trials = 1
    trainsetdir = '/home/aapostolakis/Documents/ffpdata/newcrossval/datasets/'
    testsetdir = '/home/aapostolakis/Documents/ffpdata/newcrossval/'
    #testsets = [{'training': ['*2010.csv'],'crossval': ['may*2010*', 'april*2010*']},
    #            {'training':['*2010.csv','*2011.csv'], 'crossval':['april*2011*']}]
    #testsets = [{'training': ['*features_norm.csv'],'crossval': ['may*2010*small.csv', 'april*2010*small.csv']},
    #            {'training': ['*features_norm.csv'],'crossval':['april*2011*small.csv']}]
    testsets = [{'training': ['*features_norm.csv'], 'crossval': ['april*2019_norm.csv', 'june*2019_norm.csv']},
                {'training': ['*features_norm.csv'], 'crossval': ['august*2018_norm.csv']}]

    calc_train_metrics = True
    opt_targets = ['auc', 'hybrid2', 'hybrid5', 'NH2', 'NH5', 'NH10']
    auc_thressholds=30
    #modeltype = 'tf'
    modeltype = 'sk'
    cvrownum = 1000000
    filedesc = 'NN'
    writescores=True
    debug=True
    #resdir='results/hyperopt'
    resdir='/home/aapostolakis/Documents/ffpdata/results/aris/'
    testmetrics = ['auc', 'hybrid2', 'hybrid5', 'NH2', 'NH5', 'NH10']
    cvrespattern = '*2018only*'
    hypalgo='tpe'
    filters = [{'column': 'params', 'operator': 'contains', 'value': '200'}]
    filters = ["df_flt['params'].str.contains('200')"]
    return testsets, space, testspace, cvrespattern, filters, max_trials, hypalgo, calc_train_metrics, opt_targets, trainsetdir, testsetdir, auc_thressholds,\
           modeltype, cvrownum, filedesc, runmode, writescores, resdir, debug


