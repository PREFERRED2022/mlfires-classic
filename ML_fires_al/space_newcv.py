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
             'dropout': hp.choice('dropout',[True,False]),
             #'class_weights': hp.choice('class_weights', [[1, 5],[1, 10], [1, 50], [1, 1]])
             #'class_weights': hp.choice('class_weights', [{0:1, 1:5}, {0:1, 1:10}, {0:1, 1:50}, {0:1, 1:1}]),
             'class_weights': hp.choice('class_weights', [{0:1, 1:1}, {0:2,1:3}, {0:3,1:7}, {0:1,1:4}, {0:1,1:9}, {0:1, 1:25}, {0:1, 1:50}, {0:1, 1:100} , {0:1, 1:200}]),
             #'feature_drop': hp.choice('feature_drop',['','bin','DIR','COR']),
             #'feature_drop': hp.choice('feature_drop', [[],['_dir_'],['aspect'], ['aspect', '_dir_']]),
             'feature_drop': hp.choice('feature_drop', [['wkd', 'month','f81','frequency','x','y']]),
             'max_epochs': hp.choice('max_epochs', [20]),
             #'metric': hp.choice('metric',['accuracy', 'sparse'])
             #'metric': hp.choice('metric', ['tn'])
             'metric': hp.choice('metric',['accuracy'])

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
    '''
    max_trials = 1
    trainsetdir = '/home/aapos/Documents/newcrossval/datasets/'
    testsetdir = '/home/aapos/Documents/newcrossval/'
    #testsets = [{'training': ['*2010.csv'],'crossval': ['may*2010*', 'april*2010*']},
    #            {'training':['*2010.csv','*2011.csv'], 'crossval':['april*2011*']}]
    testsets = [{'training': ['*features_norm.csv'],'crossval': ['may*2010*small.csv', 'april*2010*small.csv']},
                {'training': ['*features_norm.csv'],'crossval':['april*2011*small.csv']}]

    calc_train_metrics = True
    #opt_targets = ['hybrid1 val', 'hybrid2 val', 'f1-score 1 val.', 'auc val.', 'recall 1 val.']
    opt_targets = ['hybrid1 val']
    auc_thressholds=30
    modeltype = 'tensorflow'
    #modeltype = 'sklearn'
    cvrownum = 5000
    filedesc = 'NN'
    valst = 'val.'
    #valst = 'test'
    return testsets, space, max_trials, calc_train_metrics, opt_targets, trainsetdir, testsetdir, auc_thressholds,\
           modeltype, cvrownum, filedesc, valst, True

