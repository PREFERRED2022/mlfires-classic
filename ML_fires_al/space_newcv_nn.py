from hyperopt import hp

def create_space():
    problemfeat = ['dir_max', 'dom_dir','month', 'wkd']
    space = {'n_internal_layers': hp.choice('n_internal_layers',
                [
                    #(0, {'layer_1_0_nodes': hp.quniform('layer_1_0_nodes', 50, 350, 50)}),
                    #(1, {'layer_1_1_nodes': hp.quniform('layer_1_1_nodes', 50, 350, 50), 'layer_2_1_nodes': hp.quniform('layer_2_1_nodes', 10, 350, 50)}),
                    #(2, {'layer_1_2_nodes': hp.quniform('layer_1_2_nodes', 50, 350, 50), 'layer_2_2_nodes': hp.quniform('layer_2_2_nodes', 10, 350, 50),
                    #     'layer_3_2_nodes': hp.quniform('layer_3_2_nodes', 50, 350, 50)})

                    (0, {'layer_1_0_nodes': hp.quniform('layer_1_0_w_nodes', 100, 2100, 100)}),
                    (1, {'layer_1_1_nodes': hp.quniform('layer_1_1_w_nodes', 100, 2100, 100), 'layer_2_1_nodes': hp.quniform('layer_2_1_w_nodes', 100, 2100, 100)}),
                    (2, {'layer_1_2_nodes': hp.quniform('layer_1_2_w_nodes', 100, 2100, 100), 'layer_2_2_nodes': hp.quniform('layer_2_2_w_nodes', 100, 2100, 100),
                         'layer_3_2_nodes': hp.quniform('layer_3_2_w_nodes', 100, 2100, 100)}),
                    (3, {'layer_1_3_nodes': hp.quniform('layer_1_3_w_nodes', 100, 2100, 100), 'layer_2_3_nodes': hp.quniform('layer_2_3_w_nodes', 100, 2100, 100),
                         'layer_3_3_nodes': hp.quniform('layer_3_3_w_nodes', 100, 2100, 100), 'layer_4_3_nodes': hp.quniform('layer_4_3_w_nodes', 100, 2100, 100)}),

                    (0, {'layer_1_0_nodes': hp.quniform('layer_1_0_nodes', 10, 100, 10)}),
                    (1, {'layer_1_1_nodes': hp.quniform('layer_1_1_nodes', 10, 100, 10), 'layer_2_1_nodes': hp.quniform('layer_2_1_nodes', 10, 100, 10)}),
                    (2, {'layer_1_2_nodes': hp.quniform('layer_1_2_nodes', 10, 100, 10), 'layer_2_2_nodes': hp.quniform('layer_2_2_nodes', 10, 100, 10),
                         'layer_3_2_nodes': hp.quniform('layer_3_2_nodes', 10, 100, 10)}),
                    (3, {'layer_1_3_nodes': hp.quniform('layer_1_3_nodes', 10, 100, 10), 'layer_2_3_nodes': hp.quniform('layer_2_3_nodes', 10, 100, 10),
                         'layer_3_3_nodes': hp.quniform('layer_3_3_nodes', 10, 100, 10), 'layer_4_3_nodes': hp.quniform('layer_4_3_nodes', 10, 100, 10)}),
                    (4, {'layer_1_4_nodes': hp.quniform('layer_1_4_nodes', 10, 100, 10), 'layer_2_4_nodes': hp.quniform('layer_2_4_nodes', 10, 100, 10),
                         'layer_3_4_nodes': hp.quniform('layer_3_4_nodes', 10, 100, 10), 'layer_4_4_nodes': hp.quniform('layer_4_4_nodes', 10, 100, 10),
                         'layer_5_4_nodes': hp.quniform('layer_5_4_nodes', 10, 100, 10)})
                ]
                ),
             #'dropout': hp.choice('dropout',[0.1, 0.2, 0.3]),
             'dropout': hp.choice('dropout',[None, 0.1, 0.2, 0.3]),
             #'dropout': hp.choice('dropout',[None]),
             'class_weights': hp.choice('class_weights', [{0:1, 1:1}, {0:1, 1:2}, {0:2,1:3}, {0:1, 1:5}, {0:1, 1:10}]),
             #'class_weights': hp.choice('class_weights', [{0:1, 1:1}, {0:2,1:3}, {0:3,1:7}, {0:1,1:4}, {0:1,1:9}, {0:1, 1:25}, {0:1, 1:50}, {0:1, 1:100} , {0:1, 1:200}]),
             #'feature_drop': hp.choice('feature_drop',[problemfeat]),
             'feature_drop': hp.choice('feature_drop',[[]]),
             'max_epochs': hp.choice('max_epochs', [2000]),
             #'metric': hp.choice('metric',['accuracy', 'sparse'])
             #'metric': hp.choice('metric', ['tn'])
             'metric': hp.choice('metric',['accuracy']),
             'optimizer': hp.choice('optimizer',[{'name': 'Adam','adam_params':hp.choice('adam_params',[None])}]),
             'ES_monitor':hp.choice('ES_monitor', ['loss']),#'val_loss','loss'
             'ES_patience':hp.choice('ES_patience', [10]),
             'ES_mindelta':hp.choice('ES_mindelta', [0.0001]),
             'batch_size':hp.choice('batch_size', [512])
             }

    runmode = "val."
    max_trials = 300
    #trainsetdir = '/work2/pa21/sgirtsou/production/datasets/hard_cosine_similarity'
    trainsetdir = '/work2/pa21/sgirtsou/production/datasets/randomnofire'
    testsetdir = '/work2/pa21/sgirtsou/production'
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
    calc_train_metrics = True
    opt_targets = ['NH2', 'NH5', 'hybrid2', 'hybrid5', 'NH10', 'auc']
    aucthress=30
    debug = True
    #modeltype = 'sk'
    modeltype = 'tf'
    class0_headrows = 1000000
    filespec = "NN_nodropfeat_1M_tpe"
    writescores = False
    resdir = '/work2/pa21/sgirtsou/production/results/newcv/nn'
    suggestalgo = 'tpe'
    return testsets, space, None, None, [], max_trials, suggestalgo, calc_train_metrics, opt_targets, trainsetdir, testsetdir,\
           aucthress, modeltype, class0_headrows, filespec, runmode, writescores, resdir, debug
