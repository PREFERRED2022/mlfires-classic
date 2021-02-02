from hyperopt import hp
import os

def create_space():
    #space = [{'n_internal_layers': [0, {'layer_1_0_nodes': 50}], 'class_weights': {0:3,1:7}, 'feature_drop':['aspect'], 'metric': 'accuracy', 'dropout': False}]
    space = [{'class_weights': {0: 2, 1: 3}, 'dropout': True, 'feature_drop': ('_dir_', 'aspect'), 'metric': 'accuracy',
     'n_internal_layers': (1, {'layer_1_1_nodes': 500, 'layer_2_1_nodes': 500})}]
    space = [{'name':'best auc', 'class_weights': {0: 1, 1: 1}, 'dropout': False, 'feature_drop': (), 'metric': 'accuracy', 'n_internal_layers': (0, {'layer_1_0_nodes': 200})},
             {'class_weights': {0: 3, 1: 7}, 'dropout': True, 'feature_drop': (), 'metric': 'accuracy', 'n_internal_layers': (0, {'layer_1_0_nodes': 500})},]
             #{'class_weights': {0: 2, 1: 3}, 'dropout': False, 'feature_drop': (), 'metric': 'accuracy', 'n_internal_layers': (0, {'layer_1_0_nodes': 50})},
             #{'class_weights': {0: 3, 1: 7}, 'dropout': True, 'feature_drop': (), 'metric': 'accuracy', 'n_internal_layers': (0, {'layer_1_0_nodes': 200})},
             #{'class_weights': {0: 3, 1: 7}, 'dropout': True, 'feature_drop': (), 'metric': 'accuracy', 'n_internal_layers': (0, {'layer_1_0_nodes': 50})},
             #{'class_weights': {0: 2, 1: 3}, 'dropout': True, 'feature_drop': ('_dir_', 'aspect'), 'metric': 'accuracy', 'n_internal_layers': (1, {'layer_1_1_nodes': 500, 'layer_2_1_nodes': 500})},
             #{'class_weights': {0: 2, 1: 3}, 'dropout': True, 'feature_drop': ('_dir_', 'aspect'), 'metric': 'accuracy', 'n_internal_layers': (1, {'layer_1_1_nodes': 500, 'layer_2_1_nodes': 500})},
             #{'class_weights': {0: 2, 1: 3}, 'dropout': True, 'feature_drop': ('_dir_', 'aspect'), 'metric': 'accuracy', 'n_internal_layers': (1, {'layer_1_1_nodes': 500, 'layer_2_1_nodes': 500})},
             #{'class_weights': {0: 3, 1: 7}, 'dropout': False, 'feature_drop': ('_dir_',), 'metric': 'accuracy', 'n_internal_layers': (0, {'layer_1_0_nodes': 500})},
             #{'class_weights': {0: 1, 1: 25}, 'dropout': True, 'feature_drop': ('_dir_',), 'metric': 'accuracy', 'n_internal_layers': (0, {'layer_1_0_nodes': 50})}]


    #space = {'n_internal_layers': [0, {'layer_1_0_nodes': 50}], 'class_weights': {0:1,1:25}, 'feature_drop':['_dir_'], 'metric': 'accuracy' }

    max_epochs = 2000
    #dsfile = 'dataset_1_10_corine_level2_onehotenc.csv'
    dsfile = 'dataset_corine_level2_onehotenc.csv'
    dstestdir = '/home/sgirtsou/Documents/test_datasets_19/test_datasets_2019_dummies/'
    #dstestmonths = ['june', 'july', 'august', 'september']
    dstestmonths = ['june']
    #dstestfiles = [os.path.join(dstestdir, m+'_2019_dataset_fire_sh_dummies_clean_right_col.csv') for m in dstestmonths]
    dstestfiles = [os.path.join(dstestdir, m+'_test.csv') for m in dstestmonths]
    checkunnorm = True
    savescores = True

    return dstestfiles, dsfile, space, max_epochs, checkunnorm, savescores, 8

