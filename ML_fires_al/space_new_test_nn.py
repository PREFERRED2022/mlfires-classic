def create_space():
    space = None
    max_trials = 1
    trainsetdir = '/mnt/nvme2tb/ffp/datasets/'
    #testsetdir = '/work2/pa21/sgirtsou/production'
    #testsetdir = '/users/pa21/sgirtsou/production/2020'
    tyear = '2019'
    testsetdir = '/mnt/nvme2tb/ffp/datasets/test'
    #runmode = 'val.'
    runmode = 'test'


    #k-fold best
    testspace = { 'NH5 %s'%runmode:
                  [{'params':
                   {'ES_mindelta': 0.0001, 'ES_monitor': 'loss', 'ES_patience': 10, 'batch_size': 512, 'class_weights': {0: 1, 1: 10}, 'dropout': 0.3, 
                    'feature_drop': ('dir_max', 'dom_dir', 'month', 'wkd', 'res_max' ), 'max_epochs': 2000, 'metric': 'accuracy', 
                    'n_internal_layers': (0, {'layer_1_0_nodes': 400.0}), 'optimizer': {'adam_params': None, 'name': 'Adam'}}},
                   {'params':
                   {'ES_mindelta': 0.0001, 'ES_monitor': 'loss', 'ES_patience': 10, 'batch_size': 512, 'class_weights': {0: 1, 1: 10}, 'dropout': None,
                    'feature_drop': ('dir_max', 'dom_dir', 'month', 'wkd', 'res_max'), 'max_epochs': 2000, 'metric': 'accuracy',
                    'n_internal_layers': (0, {'layer_1_0_nodes': 50.0}), 'optimizer': {'adam_params': None, 'name': 'Adam'}}}
                  ],
                  'NH10 %s'%runmode:
                  [{'params':
                   {'ES_mindelta': 0.0001, 'ES_monitor': 'loss', 'ES_patience': 10, 'batch_size': 512, 'class_weights': {0: 1, 1: 50}, 'dropout': 0.3,
                    'feature_drop': ('dir_max', 'dom_dir', 'month', 'wkd', 'res_max'), 'max_epochs': 2000, 'metric': 'accuracy',
                    'n_internal_layers': (0, {'layer_1_0_nodes': 900.0}), 'optimizer': {'adam_params': None, 'name': 'Adam'}}},
                   {'params':
                   {'ES_mindelta': 0.0001, 'ES_monitor': 'loss', 'ES_patience': 10, 'batch_size': 512, 'class_weights': {0: 1, 1: 50}, 'dropout': None,
                    'feature_drop': ('dir_max', 'dom_dir', 'month', 'wkd', 'res_max'), 'max_epochs': 2000, 'metric': 'accuracy',
                    'n_internal_layers': (0, {'layer_1_0_nodes': 100.0}), 'optimizer': {'adam_params': None, 'name': 'Adam'}}}
                  ],
                  'NH2 %s'%runmode:
                  [{'params':
                   {'ES_mindelta': 0.0001, 'ES_monitor': 'loss', 'ES_patience': 10, 'batch_size': 512, 'class_weights': {0: 1, 1: 5}, 'dropout': 0.3,
                    'feature_drop': ('dir_max', 'dom_dir', 'month', 'wkd', 'res_max'), 'max_epochs': 2000, 'metric': 'accuracy',
                    'n_internal_layers': (0, {'layer_1_0_nodes': 200.0}), 'optimizer': {'adam_params': None, 'name': 'Adam'}}},
                  ],
                  'hybrid5 %s'%runmode:
                  [{'params':
                   {'ES_mindelta': 0.0001, 'ES_monitor': 'loss', 'ES_patience': 10, 'batch_size': 512, 'class_weights': {0: 1, 1: 5}, 'dropout': 0.3,
                    'feature_drop': ('dir_max', 'dom_dir', 'month', 'wkd', 'res_max'), 'max_epochs': 2000, 'metric': 'accuracy',
                    'n_internal_layers': (0, {'layer_1_0_nodes': 200.0}), 'optimizer': {'adam_params': None, 'name': 'Adam'}}},
                   {'params':
                   {'ES_mindelta': 0.0001, 'ES_monitor': 'loss', 'ES_patience': 10, 'batch_size': 512, 'class_weights': {0: 1, 1: 5}, 'dropout': None,
                    'feature_drop': ('dir_max', 'dom_dir', 'month', 'wkd', 'res_max'), 'max_epochs': 2000, 'metric': 'accuracy',
                    'n_internal_layers': (0, {'layer_1_0_nodes': 50.0}), 'optimizer': {'adam_params': None, 'name': 'Adam'}}}
                  ]
                }
    
    '''
    #Alt CV best
    testspace = { 'auc %s'%runmode:
                  [{'params':
                   {'ES_mindelta': 0.0001, 'ES_monitor': 'loss', 'ES_patience': 10, 'batch_size': 512, 'class_weights': {0: 1, 1: 10}, 
                   'dropout': None, 'feature_drop': ('dir_max', 'dom_dir', 'month', 'wkd', 'res_max'), 'max_epochs': 2000, 'metric': 'accuracy', 
                   'n_internal_layers': (0, {'layer_1_0_nodes': 100.0}), 'optimizer': {'adam_params': None, 'name': 'Adam'}}},
                  ],
                  'hybrid5 %s'%runmode:
                  [{'params':
                   {'ES_mindelta': 0.0001, 'ES_monitor': 'loss', 'ES_patience': 10, 'batch_size': 512, 'class_weights': {0: 1, 1: 10}, 
                    'dropout': None, 'feature_drop': ('dir_max', 'dom_dir', 'month', 'wkd', 'res_max'), 'max_epochs': 2000, 'metric': 'accuracy', 
                    'n_internal_layers': (0, {'layer_1_0_nodes': 90.0}), 'optimizer': {'adam_params': None, 'name': 'Adam'}}}
                  ],
                  'NH5 %s'%runmode:
                  [{'params':
                   {'ES_mindelta': 0.0001, 'ES_monitor': 'loss', 'ES_patience': 10, 'batch_size': 512, 'class_weights': {0: 1, 1: 10}, 
                   'dropout': None, 'feature_drop': ('dir_max', 'dom_dir', 'month', 'wkd', 'res_max'), 'max_epochs': 2000, 'metric': 'accuracy', 
                   'n_internal_layers': (1, {'layer_1_1_nodes': 40.0, 'layer_2_1_nodes': 30.0}), 'optimizer': {'adam_params': None, 'name': 'Adam'}}},
                   {'params':
                   {'ES_mindelta': 0.0001, 'ES_monitor': 'loss', 'ES_patience': 10, 'batch_size': 512, 'class_weights': {0: 1, 1: 10}, 
                    'dropout': 0.2, 'feature_drop': ('dir_max', 'dom_dir', 'month', 'wkd', 'res_max'), 'max_epochs': 2000, 'metric': 'accuracy', 
                    'n_internal_layers': (0, {'layer_1_0_nodes': 200.0}), 'optimizer': {'adam_params': None, 'name': 'Adam'}}},
                  ],
                  'NH10 %s'%runmode:
                  [{'params':
                   {'ES_mindelta': 0.0001, 'ES_monitor': 'loss', 'ES_patience': 10, 'batch_size': 512, 'class_weights': {0: 1, 1: 10}, 
                    'dropout': 0.2, 'feature_drop': ('dir_max', 'dom_dir', 'month', 'wkd', 'res_max'), 'max_epochs': 2000, 'metric': 'accuracy', 
                    'n_internal_layers': (1, {'layer_1_1_nodes': 60.0, 'layer_2_1_nodes': 60.0}), 'optimizer': {'adam_params': None, 'name': 'Adam'}}},
                   {'params':
                   {'ES_mindelta': 0.0001, 'ES_monitor': 'loss', 'ES_patience': 10, 'batch_size': 512, 'class_weights': {0: 1, 1: 10}, 
                    'dropout': None, 'feature_drop': ('dir_max', 'dom_dir', 'month', 'wkd', 'res_max'), 'max_epochs': 2000, 'metric': 'accuracy', 
                    'n_internal_layers': (1, {'layer_1_1_nodes': 40.0, 'layer_2_1_nodes': 30.0}), 'optimizer': {'adam_params': None, 'name': 'Adam'}}}
                  ]

                }
    '''

    testsets = [
                #{'training':['old_random_new_feat_from_months.csv'],\
                # 'crossval':['*april_%s_norm.csv'%tyear]},
                #{'training':['old_random_new_feat_from_months.csv'],\
                # 'crossval':['*may_%s_norm.csv'%tyear]},
                #{'training':['old_random_new_feat_from_months.csv'],\
                # 'crossval':['*june_%s_norm.csv'%tyear]},
                #{'training':['old_random_new_feat_from_months.csv'],\
                # 'crossval':['*july_%s_norm.csv'%tyear]},
                #{'training':['old_random_new_feat_from_months.csv'],\
                # 'crossval':['*august_%s_norm.csv'%tyear]},
                #{'training':['old_random_new_feat_from_months.csv'],\
                # 'crossval':['*september_%s_norm.csv'%tyear]},
                {'training':['fires_new_norm.csv'],\
                 'crossval':['%s06*_norm.csv'%tyear]},
                {'training':['fires_new_norm.csv'],\
                 'crossval':['%s07*_norm.csv'%tyear]},
                {'training':['fires_new_norm.csv'],\
                 'crossval':['%s08*_norm.csv'%tyear]},
                {'training':['fires_new_norm.csv'],\
                 'crossval':['%s09*_norm.csv'%tyear]},
                #{'training':['old_random_new_feat_from_months.csv'],\
                # 'crossval':['*_%s_norm.csv'%tyear]}
                #{'training':['old_random_new_feat_from_months.csv'],\
                # 'crossval':['*_norm.csv']}
               ]

    calc_train_metrics = True
    opt_targets = ['auc', 'f1-score 1', 'hybrid1', 'hybrid2', 'hybrid5', 'NH2', 'NH5', 'NH10']
    #opt_targets = ['f1-score 1']
    #opt_targets = ['auc', 'hybrid2', 'hybrid5', 'NH2', 'NH5', 'NH10']
    aucthress=0
    debug = True
    #modeltype = 'sk'
    modeltype = 'tf'
    class0_headrows = 0
    filespec = "%s_nd"%tyear
    writescore = False
    resdir = '/mnt/nvme2tb/ffp/results/newdefCV/'
    #resdir = '/work2/pa21/sgirtsou/production/results/newcv/nn/'
    #cvrespattern = '*_dropfeat_*_mean*'
    #cvrespattern = '*_dropfeat_1M_*_mean*'
    cvrespattern = '*NN_vl*mean*'
    filters = ["df_flt['params'].str.contains(\"'dropout': None\")"]
    #filters = ["~df_flt['params'].str.contains(\"'dropout': None\")"]
    #calib = {'min_temp':-0.15, 'dom_vel': -0.40, 'mean_temp': 0.2, 'mean_dew_temp': 0.2, 'min_dew_temp':0.2 , 'rain_7days': -0.999}
    calib = {}
    #xlaflags='--xla_gpu_cuda_data_dir=/usr/lib/cuda'
    return testsets, space, testspace, cvrespattern, filters, max_trials, calc_train_metrics, opt_targets, trainsetdir, testsetdir,\
           aucthress, modeltype, filespec, runmode, writescore, resdir, calib, debug

