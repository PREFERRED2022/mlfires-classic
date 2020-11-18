from hyperopt import hp

def create_space():
    space = {'n_internal_layers': hp.choice('n_internal_layers',
                [
                    #(0, {'layer_1_0_nodes': hp.quniform('layer_1_0_nodes', 10, 310, 50)}),
                    #(1, {'layer_1_1_nodes': hp.quniform('layer_1_1_nodes', 10, 310, 50), 'layer_2_1_nodes': hp.quniform('layer_2_1_nodes', 10, 310, 50)}),
                    #(2, {'layer_1_2_nodes': hp.quniform('layer_1_2_nodes', 10, 310, 50), 'layer_2_2_nodes': hp.quniform('layer_2_2_nodes', 10, 310, 50),
                    #     'layer_3_2_nodes': hp.quniform('layer_3_2_nodes', 10, 310, 50)})

                    (0, {'layer_1_0_nodes': hp.choice('layer_1_0_nodes', [50])}),
                    #(1, {'layer_1_1_nodes': hp.choice('layer_1_1_nodes', [500, 1000]),
                    #     'layer_2_1_nodes': hp.choice('layer_2_1_nodes', [500, 1000])}),
                    #(2, {'layer_1_2_nodes': hp.choice('layer_1_2_nodes', [500, 1000]),
                    #     'layer_2_2_nodes': hp.choice('layer_2_2_nodes', [500, 1000]),
                    #     'layer_3_2_nodes': hp.choice('layer_3_2_nodes', [500, 1000])}),
                ]
                ),
             #'class_weights': hp.choice('class_weights', [[1, 5],[1, 10], [1, 50], [1, 1]])
             #'class_weights': hp.choice('class_weights', [{0:1, 1:5}, {0:1, 1:10}, {0:1, 1:50}, {0:1, 1:1}]),
             'class_weights': hp.choice('class_weights', [{0:1, 1:1}, {0:2,1:3}, {0:3,1:7}, {0:1,1:4}, {0:1,1:9}, {0:1, 1:25}, {0:1, 1:50}, {0:1, 1:100} , {0:1, 1:200}]),
             #'feature_drop': hp.choice('feature_drop',['','bin','DIR','COR']),
             'feature_drop': hp.choice('feature_drop', ['']),
             #'metric': hp.choice('metric',['accuracy', 'sparse'])
             #'metric': hp.choice('metric', ['tn'])
             'metric': hp.choice('metric',['accuracy'])

             #'feature_drop': hp.choice('feature_drop', ['bin'])
             }
    '''
    space = {'n_internal_layers': hp.choice('n_internal_layers',
                [(0, {'layer_1_0_nodes': hp.choice('layer_1_0_nodes', [50])})]

                ),
             'class_weights': hp.choice('class_weights', [[1, 5],[1, 10], [1, 50], [1, 1]])
             }
    '''
    max_trials = 1
    max_epochs = 50
    dsfile = 'dataset_1_10.csv'
    return dsfile, space, max_trials, max_epochs

