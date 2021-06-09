from hyperopt import hp

def create_space():
    newfeatures = ['x', 'y', 'month', 'wkd', 'lst', 'dew', 'freq', 'f81']
    problemfeatures = ['month', 'wkd', 'dom_dir', 'dir_max']

    space = {'n_internal_layers': hp.choice('n_internal_layers',
                [
                    #(0, {'layer_1_0_nodes': hp.quniform('layer_1_0_nodes', 10, 310, 50)}),
                    #(1, {'layer_1_1_nodes': hp.quniform('layer_1_1_nodes', 10, 310, 50), 'layer_2_1_nodes': hp.quniform('layer_2_1_nodes', 10, 310, 50)}),
                    #(2, {'layer_1_2_nodes': hp.quniform('layer_1_2_nodes', 10, 310, 50), 'layer_2_2_nodes': hp.quniform('layer_2_2_nodes', 10, 310, 50),
                    #     'layer_3_2_nodes': hp.quniform('layer_3_2_nodes', 10, 310, 50)})

                    (0, {'layer_1_0_nodes': hp.choice('layer_1_0_nodes', [1400] )}),
                    #(1, {'layer_1_1_nodes': hp.choice('layer_1_1_nodes', [1750]),
                    #     'layer_2_1_nodes': hp.choice('layer_2_1_nodes', [200])}),
                    #(2, {'layer_1_2_nodes': hp.choice('layer_1_2_nodes', [500, 1000]),
                    #     'layer_2_2_nodes': hp.choice('layer_2_2_nodes', [500, 1000]),
                    #     'layer_3_2_nodes': hp.choice('layer_3_2_nodes', [500, 1000])}),
                ]
                ),
             'dropout': hp.choice('dropout',[None, 0.2]),
             #'class_weights': hp.choice('class_weights', [{0:1, 1:1}, {0:2,1:3}, {0:3,1:7}, {0:1,1:4}, {0:1,1:9}, {0:1, 1:25}, {0:1, 1:50}, {0:1, 1:100} , {0:1, 1:200}]),
             'class_weights': hp.choice('class_weights', [{0: 1, 1: 1}]),
             #'class_weights': {0: hp.choice('class_0_weight', [1]), 1: hp.quniform('class_1_weight', 1,100,1)},

             #'feature_drop': hp.choice('feature_drop',['','bin','DIR','COR']),
             #'feature_drop': hp.choice('feature_drop', [[], newfeatures, ['x'], ['y'], ['x', 'y'], ['month', 'wkd'],['dir_'], ['aspect'], ['corine'], \
             #                                          ['lst'], ['dew'], ['lst', 'dew'], newfeatures+['dir_','aspect', 'corine']]),
             'feature_drop': problemfeatures,
             #'metric': hp.choice('metric',['accuracy', 'sparse'])
             'metric': hp.choice('metric',['accuracy']),
             'optimizer': hp.choice('optimizer', [{'name': 'Adam', 'adam_params': None}]),
             # 'optimizer': hp.choice('optimizer',[{'name': 'Adam',
             #                                     'adam_params':hp.choice('adam_params',
             #                                      [None,{'learning_rate_adam':hp.uniform('learning_rate_adam', 0.0001, 1),\
             #                                      'beta_1':hp.uniform('beta_1', 0.0001, 1), 'beta_2':hp.uniform('beta_2', 0.0001, 1),\
             #                                      'amsgrad': hp.choice('amsgrad', [True, False])}])},
             #                                    {'name': 'SGD', 'learning_rate_SGD':hp.uniform('learning_rate_SGD', 0.0001, 1)}]),
             'max_epochs': hp.choice('max_epochs', [2000]),
             'ES_monitor':hp.choice('ES_monitor', ['loss']),#'val_loss','loss'
             'ES_patience':hp.choice('ES_patience', [10]),
             'ES_mindelta':hp.choice('ES_mindelta', [0.002]),
             'batch_size':hp.choice('batch_size', [512])
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
    
    {'n_estimators': 250, 'min_samples_split': 180, 'min_samples_leaf': 40, 'max_features': 41, 'max_depth': 20, 'criterion': 'entropy', 'class_weight': {0: 1, 1: 9}, 'bootstrap': True}
    
    space = {'algo': hp.choice('algo', ['RF']),
             'n_estimators': 250,
             #'n_estimators': hp.choice('n_estimators', [50, 100, 120, 150, 170, 200, 250, 350, 500, 750, 1000, 1400, 1500]),
             'min_samples_split': 180,
             #'min_samples_split': hp.choice('min_samples_split',
             #                               [2, 10, 50, 70, 100, 120, 150, 180, 200, 250, 400, 600, 1000, 1300, 2000]),
             'min_samples_leaf': 40,
             #'min_samples_leaf' :hp.choice('min_samples_leaf',[1, 10,30,40,50,100,120,150]),
             'criterion': 'entropy',
             #'criterion':hp.choice('criterion',["gini", "entropy"]),
             'max_features': 4,
             #'max_features':hp.quniform('max_features', 1,10,1),
             'bootstrap':True,
             #'bootstrap':hp.choice('bootstrap',[True, False]),
             'max_depth': 20,
             #'max_depth': hp.choice('max_depth', [10, 20, 100, 200, 400, 500, 700, 1000, 1200, 2000, None]),
             'feature_drop': [],
             #'feature_drop': hp.choice('feature_drop', [['wkd', 'month']]),
             'class_weights': {0: 1, 1: 9}
             #'class_weight':hp.choice('class_weight',[{0:1,1:1}, {0:1,1:9},{0:1,1:300},{0:1,1:400},{0:1,1:500},{0:1,1:1000}])
             }
    '''
    max_trials = 1
    #testsets = {'balanced':'/home/aapos/Documents/newcrossval/datasets/randomnofire/old_random_new_features_norm.csv', 'imbalanced':'dataset_1_10_corine_level2_onehotenc.csv'}
    testsets = {'balanced':'/home/aapostolakis/Documents/ffpdata/newcrossval/datasets/randomnofire/oldrandomnewfeat.csv', 'imbalanced':'dataset_1_10_corine_level2_onehotenc.csv'}
    calc_train_metrics = True
    #opt_targets = ['hybrid2 val', 'hybrid5 val', 'f1-score 1 val.', 'auc val.', 'recall 1 val.']
    opt_targets = ['auc val.']
    modeltype = 'tf'
    #modeltype = 'sk'
    description = 'NN'
    nfolds = 5
    writescores = False
    resultsfolder = 'results/hyperopt'
    return 'balanced', testsets, nfolds, space, max_trials, calc_train_metrics, opt_targets, modeltype, description,\
           writescores, resultsfolder

