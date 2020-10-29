from hyperopt import hp

def create_space():
    space = {'n_internal_layers': hp.choice('n_internal_layers',
                [
                    #(0, {'layer_1_0_nodes': hp.quniform('layer_1_0_nodes', 10, 310, 50)}),
                    #(1, {'layer_1_1_nodes': hp.quniform('layer_1_1_nodes', 10, 310, 50), 'layer_2_1_nodes': hp.quniform('layer_2_1_nodes', 10, 310, 50)}),
                    #(2, {'layer_1_2_nodes': hp.quniform('layer_1_2_nodes', 10, 310, 50), 'layer_2_2_nodes': hp.quniform('layer_2_2_nodes', 10, 310, 50),
                    #     'layer_3_2_nodes': hp.quniform('layer_3_2_nodes', 10, 310, 50)})

                    (0, {'layer_1_0_nodes': hp.choice('layer_1_0_nodes', [500, 1000])}),
                    (1, {'layer_1_1_nodes': hp.choice('layer_1_1_nodes', [500, 1000]),
                         'layer_2_1_nodes': hp.choice('layer_2_1_nodes', [500, 1000])}),
                    (2, {'layer_1_2_nodes': hp.choice('layer_1_2_nodes', [500, 1000]),
                         'layer_2_2_nodes': hp.choice('layer_2_2_nodes', [500, 1000]),
                         'layer_3_2_nodes': hp.choice('layer_3_2_nodes', [500, 1000])}),
                ]
                ),
             'class_weights': hp.choice('class_weights', [[1, 5],[1, 10], [1, 50], [1, 1]])
             }
    '''
    space = {'n_internal_layers': hp.choice('n_internal_layers',
                [(0, {'layer_1_0_nodes': hp.choice('layer_1_0_nodes', [50])})]

                ),
             'class_weights': hp.choice('class_weights', [[1, 5],[1, 10], [1, 50], [1, 1]])
             }
    '''
    return space
