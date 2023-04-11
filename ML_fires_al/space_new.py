from hyperopt import hp,tpe

def create_space():
    newfeatures = ['x', 'y', 'month', 'wkd', 'lst', 'dew', 'freq', 'f81']
    problemfeatures = ['month', 'weekday', 'dom_dir', 'dir_max']

    space = {'n_internal_layers': hp.choice('n_internal_layers',
                [
                    (0, {'layer_1_0_nodes': hp.quniform('layer_1_0_w_nodes', 100, 2100, 100)}),
                    (1, {'layer_1_1_nodes': hp.quniform('layer_1_1_w_nodes', 100, 2100, 100),
                         'layer_2_1_nodes': hp.quniform('layer_2_1_w_nodes', 100, 2100, 100)}),
                    (2, {'layer_1_2_nodes': hp.quniform('layer_1_2_w_nodes', 100, 2100, 100),
                         'layer_2_2_nodes': hp.quniform('layer_2_2_w_nodes', 100, 2100, 100),
                         'layer_3_2_nodes': hp.quniform('layer_3_2_w_nodes', 100, 2100, 100)}),
                    (3, {'layer_1_3_nodes': hp.quniform('layer_1_3_w_nodes', 100, 2100, 100),
                         'layer_2_3_nodes': hp.quniform('layer_2_3_w_nodes', 100, 2100, 100),
                         'layer_3_3_nodes': hp.quniform('layer_3_3_w_nodes', 100, 2100, 100),
                         'layer_4_3_nodes': hp.quniform('layer_4_3_w_nodes', 100, 2100, 100)}),

                    (0, {'layer_1_0_nodes': hp.quniform('layer_1_0_nodes', 10, 100, 10)}),
                    (1, {'layer_1_1_nodes': hp.quniform('layer_1_1_nodes', 10, 100, 10),
                         'layer_2_1_nodes': hp.quniform('layer_2_1_nodes', 10, 100, 10)}),
                    (2, {'layer_1_2_nodes': hp.quniform('layer_1_2_nodes', 10, 100, 10),
                         'layer_2_2_nodes': hp.quniform('layer_2_2_nodes', 10, 100, 10),
                         'layer_3_2_nodes': hp.quniform('layer_3_2_nodes', 10, 100, 10)}),
                    (3, {'layer_1_3_nodes': hp.quniform('layer_1_3_nodes', 10, 100, 10),
                         'layer_2_3_nodes': hp.quniform('layer_2_3_nodes', 10, 100, 10),
                         'layer_3_3_nodes': hp.quniform('layer_3_3_nodes', 10, 100, 10),
                         'layer_4_3_nodes': hp.quniform('layer_4_3_nodes', 10, 100, 10)}),
                    (4, {'layer_1_4_nodes': hp.quniform('layer_1_4_nodes', 10, 100, 10),
                         'layer_2_4_nodes': hp.quniform('layer_2_4_nodes', 10, 100, 10),
                         'layer_3_4_nodes': hp.quniform('layer_3_4_nodes', 10, 100, 10),
                         'layer_4_4_nodes': hp.quniform('layer_4_4_nodes', 10, 100, 10),
                         'layer_5_4_nodes': hp.quniform('layer_5_4_nodes', 10, 100, 10)})
                ]
                ),
             'dropout': hp.choice('dropout',[None, 0.1, 0.2, 0.3]),
             'class_weights': hp.choice('class_weights', [{0: 1, 1: 1}, {0: 1, 1: 2}, {0: 2, 1: 3}, {0: 1, 1: 5}, {0: 1, 1: 10}]),
             'feature_drop': hp.choice('feature_drop',[problemfeatures, []]),
             'metric': hp.choice('metric',['accuracy']),
             'optimizer': hp.choice('optimizer', [{'name': 'Adam', 'adam_params': None}]),
             'max_epochs': hp.choice('max_epochs', [2000]),
             'ES_monitor':hp.choice('ES_monitor', ['loss']),#'val_loss','loss'
             'ES_patience':hp.choice('ES_patience', [10]),
             'ES_mindelta':hp.choice('ES_mindelta', [0.002]),
             'batch_size': hp.choice('batch_size', [512]),
             }

    max_trials = 500
    #testsets = {'balanced':'/home/aapos/Documents/newcrossval/datasets/randomnofire/old_random_new_features_norm.csv', 'imbalanced':'dataset_1_10_corine_level2_onehotenc.csv'}
    testsets = {'balanced':'/data2/ffp/datasets/trainingsets/newfull/traindataset_new_norm.csv', 'imbalanced':'dataset_1_10_corine_level2_onehotenc.csv'}
    calc_train_metrics = True
    #opt_targets = ['hybrid2 val', 'hybrid5 val', 'f1-score 1 val.', 'auc val.', 'recall 1 val.']
    #opt_targets = ['auc val.']
    #opt_targets = ['NH2', 'NH5', 'hybrid2', 'hybrid5', 'NH10', 'auc']
    opt_targets = ['NH5']
    modeltype = 'tf'
    #modeltype = 'sk'
    description = 'NN'
    nfolds = 5
    writescores = False
    resultsfolder = '/data2/ffp/results/newdefCV'
    #hypalgo = 'random'
    hypalgo = 'tpe'
    return 'balanced', testsets, nfolds, space, max_trials, hypalgo, calc_train_metrics, opt_targets, modeltype, description,\
           writescores, resultsfolder

