from hyperopt import hp,tpe

def create_space():
    newfeatures = ['x', 'y', 'month', 'wkd', 'lst', 'dew', 'freq', 'f81']
    dropfeat1 = ['month', 'weekday', 'dom_dir', 'dir_max']+['corine_%d' % i for i in range(1, 10)]
    dropfeat2 = ['month', 'weekday', 'dom_dir', 'dir_max']+['corine_gr%d' % i for i in range(1, 10)]
    dropfeat3 = ['month', 'weekday', 'dom_dir', 'dir_max', 'pop']+['corine_%d' % i for i in range(1, 10)]
    dropfeat4 = ['month', 'weekday', 'dom_dir', 'dir_max', 'pop']+['corine_gr%d' % i for i in range(1, 10)]
    dropfeat5 = ['month', 'weekday', 'dom_dir', 'dir_max'] + ['corine_gr%d' % i for i in range(1, 10)] + \
                ['corine_gr%d' % i for i in range(1, 10)]

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
             'dropout': hp.choice('dropout',[None, 0.3]),
             'class_weights': hp.choice('class_weights', [{0: 1, 1: 1}, {0: 1, 1: 2}, {0: 2, 1: 3}, {0: 1, 1: 5}, {0: 1, 1: 10}]),
             #'feature_drop': hp.choice('feature_drop',[problemfeatures, []]),
             'feature_drop': hp.choice('feature_drop',[dropfeat1,dropfeat2,dropfeat3,dropfeat4]),
             #'feature_drop': hp.choice('feature_drop', [dropfeat1, dropfeat2, dropfeat5]),
             'metric': hp.choice('metric',['accuracy']),
             'optimizer': hp.choice('optimizer', [{'name': 'Adam', 'adam_params': None}]),
             'max_epochs': hp.choice('max_epochs', [2000]),
             'ES_monitor':hp.choice('ES_monitor', ['val_loss']),#'val_loss','loss'
             'ES_patience':hp.choice('ES_patience', [10]),
             'ES_mindelta':hp.choice('ES_mindelta', [0.002]),
             'batch_size': hp.choice('batch_size', [512]),
             }

    max_trials = 200
    #testsets = {'balanced':'/home/aapos/Documents/newcrossval/datasets/randomnofire/old_random_new_features_norm.csv', 'imbalanced':'dataset_1_10_corine_level2_onehotenc.csv'}
    testsets = {'balanced':'/mnt/nvme2tb/ffp/datasets/train/train_new_sample_1_2_norm.csv', \
                'imbalanced':'dataset_1_10_corine_level2_onehotenc.csv'}
    calc_train_metrics = True
    #opt_targets = ['auc val.']
    #opt_targets = ['NH2', 'NH5', 'hybrid2', 'hybrid5', 'NH10', 'auc', 'hybrid1', 'f1-score 1']
    opt_targets = ['hybrid2']
    modeltype = 'tf'
    #modeltype = 'sk'
    description = 'NN_ns'
    nfolds = 5
    writescores = False
    resultsfolder = '/mnt/nvme2tb/ffp/results/newdefCV'
    #hypalgo = 'random'
    hypalgo = 'tpe'
    gpumb=0
    return 'balanced', testsets, nfolds, space, max_trials, hypalgo, calc_train_metrics, opt_targets, modeltype, description,\
           writescores, resultsfolder, gpumb

