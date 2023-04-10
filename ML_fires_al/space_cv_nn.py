from hyperopt import hp

def create_space():
    newfeatures = ['xpos', 'ypos', 'month', 'wkd', 'lst', 'dew', 'freq', 'f81']
    categorical = ['dir_max', 'dom_dir','month', 'wkd', 'corine']
    problemfeat = ['dir_max', 'dom_dir','month', 'wkd']
    goodnewfeat = ['xpos', 'ypos','lst', 'dew', 'freq', 'f81']
    space = {'n_internal_layers': hp.choice('n_internal_layers',
                [
                    #(0, {'layer_1_0_nodes': hp.quniform('layer_1_0_nodes', 100, 2100, 100)}),
                    #(1, {'layer_1_1_nodes': hp.quniform('layer_1_1_nodes', 100, 2100, 100), 'layer_2_1_nodes': hp.quniform('layer_2_1_nodes', 100, 2100, 100)}),
                    #(2, {'layer_1_2_nodes': hp.quniform('layer_1_2_nodes', 100, 2100, 100), 'layer_2_2_nodes': hp.quniform('layer_2_2_nodes', 100, 2100, 100),
                    #     'layer_3_2_nodes': hp.quniform('layer_3_2_nodes', 100, 2100, 100)}),
                    #(3, {'layer_1_3_nodes': hp.quniform('layer_1_3_nodes', 100, 2100, 100), 'layer_2_3_nodes': hp.quniform('layer_2_3_nodes', 100, 2100, 100),
                    #     'layer_3_3_nodes': hp.quniform('layer_3_3_nodes', 100, 2100, 100), 'layer_4_3_nodes': hp.quniform('layer_4_3_nodes', 100, 2100, 100)}),
                    #(0, {'layer_1_0_nodes': hp.choice('layer_1_0_nodes', [hp.quniform('layer_1_0_small', 10, 100, 10), hp.quniform('layer_1_0_big', 100, 2100, 100)])}),
                    (0, {'layer_1_0_nodes': 100})
                    #(0, {'layer_1_0_nodes': hp.quniform('layer_1_0_nodes', 10, 100, 10)}),
                    #(1, {'layer_1_1_nodes': hp.quniform('layer_1_1_nodes', 10, 100, 10), 'layer_2_1_nodes': hp.quniform('layer_2_1_nodes', 10, 100, 10)}),
                    #(2, {'layer_1_2_nodes': hp.quniform('layer_1_2_nodes', 10, 100, 10), 'layer_2_2_nodes': hp.quniform('layer_2_2_nodes', 10, 100, 10),
                    #     'layer_3_2_nodes': hp.quniform('layer_3_2_nodes', 10, 100, 10)}),
                    #(3, {'layer_1_3_nodes': hp.quniform('layer_1_3_nodes', 10, 100, 10), 'layer_2_3_nodes': hp.quniform('layer_2_3_nodes', 10, 100, 10),
                    #     'layer_3_3_nodes': hp.quniform('layer_3_3_nodes', 10, 100, 10), 'layer_4_3_nodes': hp.quniform('layer_4_3_nodes', 10, 100, 10)}),
                    #(4, {'layer_1_4_nodes': hp.quniform('layer_1_4_nodes', 10, 100, 10), 'layer_2_4_nodes': hp.quniform('layer_2_4_nodes', 10, 100, 10),
                    #     'layer_3_4_nodes': hp.quniform('layer_3_4_nodes', 10, 100, 10), 'layer_4_4_nodes': hp.quniform('layer_4_4_nodes', 10, 100, 10),
                    #     'layer_5_4_nodes': hp.quniform('layer_5_4_nodes', 10, 100, 10)})
                    #(0, {'layer_1_0_nodes': hp.choice('layer_1_0_nodes', [200] )}),
                    #(1, {'layer_1_1_nodes': hp.choice('layer_1_1_nodes', [500, 1000]),
                    #     'layer_2_1_nodes': hp.choice('layer_2_1_nodes', [500, 1000])}),
                    #(2, {'layer_1_2_nodes': hp.choice('layer_1_2_nodes', [500]),
                    #     'layer_2_2_nodes': hp.choice('layer_2_2_nodes', [500]),
                    #     'layer_3_2_nodes': hp.choice('layer_3_2_nodes', [500])}),
                ]
                ),
             #'dropout': hp.choice('dropout',[0.1, 0.2, 0.3]),
             'dropout': hp.choice('dropout',[None]),
             #'dropout': hp.choice('dropout',[None, 0.1, 0.2, 0.3]),
             #'class_weights': hp.choice('class_weights', [{0:1, 1:1}]),
             #'class_weights': {0: hp.choice('class_0_weight', [1]), 1: hp.quniform('class_1_weight', 1,100,10)},
             #'class_weights': hp.choice('class_weights', [{0:1, 1:1}, {0:2,1:3}, {0:3,1:7}, {0:1,1:4}, {0:1,1:9}, {0:1, 1:25}, {0:1, 1:50}, {0:1, 1:100} , {0:1, 1:200}]),
             'class_weights': hp.choice('class_weights', [{0: 1, 1: 1}, {0:2,1:3}, {0:1,1:2}, {0:1,1:5}, {0:1,1:10}, {0:1,1:50}]),
             #'feature_drop': hp.choice('feature_drop',[[]]),
             'feature_drop': hp.choice('feature_drop',[problemfeat]),
             #'feature_drop': hp.choice('feature_drop', [[], newfeatures, problemfeat, ['wkd'], ['month'], ['dir_max'], ['dom_dir'], ['month', 'wkd'], ['dir'],\
             #                                          list(set(newfeatures)-set(['dew'])), list(set(newfeatures)-set(['lst'])),
             #                                          list(set(newfeatures)-set(['freq'])), list(set(newfeatures)-set(['f81'])),
             #                                          list(set(newfeatures)-set(['xpos'])), list(set(newfeatures)-set(['ypos'])),
             #                                          list(set(newfeatures)-set(['xpos','ypos'])), list(set(newfeatures)-set(['dew','lst'])),
             #                                          list(set(newfeatures)-set(['freq','f81'])), problemfeat+['aspect']
             #                                          ]),
             #'metric': hp.choice('metric',['accuracy', 'sparse'])
             #'metric': hp.choice('metric', ['tn'])
             'metric': hp.choice('metric',['accuracy']),
             'max_epochs':hp.choice('max_epochs',[2000]),
             #'optimizer': hp.choice('optimizer',[{'name': 'Adam',
             #                                     'adam_params':hp.choice('adam_params',
             #                                      [None,{'learning_rate_adam':hp.uniform('learning_rate_adam', 0.0001, 0.1),\
             #                                      'beta_1':hp.uniform('beta_1', 0.9, 1), 'beta_2':hp.uniform('beta_2', 0.9, 1),\
             #                                      'amsgrad': hp.choice('amsgrad', [True, False])}])},
             #                                    {'name': 'SGD', 'learning_rate_SGD':hp.uniform('learning_rate_SGD', 0.0001, 0.1)}]),
             'optimizer': hp.choice('optimizer',[{'name': 'Adam','adam_params':hp.choice('adam_params',[None])}]),
             'ES_monitor':hp.choice('ES_monitor', ['loss']),#'val_loss','loss'
             'ES_patience':hp.choice('ES_patience', [2, 5, 10, 20]),
             #'ES_mindelta': hp.uniform('ES_mindelta', 0.0001, 0.01]),
             'ES_mindelta':hp.choice('ES_mindelta', [0.002]),
             'batch_size':hp.choice('batch_size', [512])
             }
    max_trials = 300
    testsets = {'balanced':'/work2/pa21/sgirtsou/production/datasets/randomnofire/old_random_new_feat_from_months.csv', 'imbalanced':'dataset_1_10_corine_level2_onehotenc.csv'}
    #testsets = {'balanced':'/work2/pa21/sgirtsou/production/datasets/randomnofire/old_random_new_features_norm.csv', 'imbalanced':'dataset_1_10_corine_level2_onehotenc.csv'}
    dstestfile = '/home/sgirtsou/Documents/test_datasets_19/test_datasets_2019_dummies/august_2019_dataset_fire_sh_dummies.csv'
    calc_train_metrics = True
    #targets = ['hybrid2', 'hybrid5', 'NH2', 'NH5', 'NH10']
    targets = ['NH5']
    valst = 'val.'
    opt_targets = ['%s %s'%(ot,valst) for ot in targets]
    modeltype='tf'
    #modeltype='sk'
    nfolds=5
    desc='NN_patience'
    writescore =  True
    resdir = '/work2/pa21/sgirtsou/production/results/kfoldcv/nn'
    suggest_algo = 'random'

    return 'balanced', testsets, nfolds, space, max_trials,  suggest_algo, calc_train_metrics, opt_targets, modeltype, desc, writescore, resdir

