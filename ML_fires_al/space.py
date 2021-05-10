from hyperopt import hp

def create_space():
    newfeatures = ['x', 'y', 'month', 'wkd', 'lst', 'dew', 'freq', 'f81']
    '''
    space = {'n_internal_layers': hp.choice('n_internal_layers',
                [
                    #(0, {'layer_1_0_nodes': hp.quniform('layer_1_0_nodes', 10, 310, 50)}),
                    #(1, {'layer_1_1_nodes': hp.quniform('layer_1_1_nodes', 10, 310, 50), 'layer_2_1_nodes': hp.quniform('layer_2_1_nodes', 10, 310, 50)}),
                    #(2, {'layer_1_2_nodes': hp.quniform('layer_1_2_nodes', 10, 310, 50), 'layer_2_2_nodes': hp.quniform('layer_2_2_nodes', 10, 310, 50),
                    #     'layer_3_2_nodes': hp.quniform('layer_3_2_nodes', 10, 310, 50)})

                    (0, {'layer_1_0_nodes': hp.choice('layer_1_0_nodes', [200] )}),
                    #(1, {'layer_1_1_nodes': hp.choice('layer_1_1_nodes', [500, 1000]),
                    #     'layer_2_1_nodes': hp.choice('layer_2_1_nodes', [500, 1000])}),
                    #(2, {'layer_1_2_nodes': hp.choice('layer_1_2_nodes', [500, 1000]),
                    #     'layer_2_2_nodes': hp.choice('layer_2_2_nodes', [500, 1000]),
                    #     'layer_3_2_nodes': hp.choice('layer_3_2_nodes', [500, 1000])}),
                ]
                ),
             'dropout': hp.choice('dropout',[False]),
             #'class_weights': hp.choice('class_weights', [[1, 5],[1, 10], [1, 50], [1, 1]])
             #'class_weights': hp.choice('class_weights', [{0:1, 1:5}, {0:1, 1:10}, {0:1, 1:50}, {0:1, 1:1}]),
             #'class_weights': hp.choice('class_weights', [{0:1, 1:1}, {0:2,1:3}, {0:3,1:7}, {0:1,1:4}, {0:1,1:9}, {0:1, 1:25}, {0:1, 1:50}, {0:1, 1:100} , {0:1, 1:200}]),
             'class_weights': hp.choice('class_weights', [{0: 1, 1: 1}]),

             #'feature_drop': hp.choice('feature_drop',['','bin','DIR','COR']),
             'feature_drop': hp.choice('feature_drop', [[], newfeatures, ['x'], ['y'], ['x', 'y'], ['month', 'wkd'],['dir_'], ['aspect'], ['corine'], \
                                                       ['lst'], ['dew'], ['lst', 'dew'], newfeatures+['dir_','aspect', 'corine']]),
             #'metric': hp.choice('metric',['accuracy', 'sparse'])
             #'metric': hp.choice('metric', ['tn'])
             'metric': hp.choice('metric',['accuracy'])

             #'feature_drop': hp.choice('feature_drop', ['bin'])
             }
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
             'feature_drop': hp.choice('feature_drop', [['wkd', 'month']]),
             'class_weight':hp.choice('class_weight',[{0:1,1:1}, {0:1,1:9},{0:1,1:300},{0:1,1:400},{0:1,1:500},{0:1,1:1000}])
             }
    '''
    space = {'algo': hp.choice('algo', ['RF']),
             'n_estimators': hp.choice('n_estimators', [120]),
             'min_samples_split': hp.choice('min_samples_split', [150]),
             'min_samples_leaf': hp.choice('min_samples_leaf', [30]),
             'criterion': hp.choice('criterion', ["entropy"]),
             'max_features': hp.choice('max_features', [1.5]),
             'bootstrap': hp.choice('bootstrap', [False]),
             'max_depth': hp.choice('max_depth', [1000]),
             'class_weight': hp.choice('class_weight', [{0: 2, 1: 8}]),
             'feature_drop': hp.choice('feature_drop', [['wkd', 'month']]),
             }
    max_trials = 1
    #dsfile = 'dataset_1_10_corine_level2_onehotenc.csv'
    #dsfile = 'dataset_corine_level2_onehotenc.csv'
    #testsets = {'balanced':'/home/aapos/Documents/newcrossval/datasets/randomnofire/old_random_new_features_norm.csv', 'imbalanced':'dataset_1_10_corine_level2_onehotenc.csv'}
    testsets = {'balanced':'/home/aapos/Documents/newcrossval/datasets/randomnofire/oldrandomnewfeat.csv', 'imbalanced':'dataset_1_10_corine_level2_onehotenc.csv'}
    dstestfile = '/home/sgirtsou/Documents/test_datasets_19/test_datasets_2019_dummies/august_2019_dataset_fire_sh_dummies.csv'
    calc_train_metrics = True
    #opt_targets = ['hybrid1 val', 'hybrid2 val', 'f1-score 1 val.', 'auc val.', 'recall 1 val.']
    opt_targets = ['auc val.']
    #modeltype = 'tensorflow'
    modeltype = 'sklearn'
    description = 'RF'
    nfolds = 5
    return 'balanced', testsets, nfolds, space, max_trials, calc_train_metrics, opt_targets, modeltype, description

