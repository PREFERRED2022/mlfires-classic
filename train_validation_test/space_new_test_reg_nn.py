def create_space():
    space = None
    max_trials = 1
    trainsetdir = '/mnt/nvme2tb/ffp/datasets/train/'
    tyear = '2019'
    testsetdir = '/mnt/nvme2tb/ffp/datasets/test/2019/'
    #runmode = 'val.'
    runmode = 'test'

    testspace = { 'NH2 %s'%runmode:
                  [{'params':
                   {'ES_mindelta': 0.002, 'ES_monitor': 'loss', 'ES_patience': 10, 'batch_size': 512, 'class_weights': {0: 1, 1: 2},
                   'dropout': None, 'feature_drop': ('month', 'weekday', 'dom_dir', 'dir_max', 'road_dens', 'pop'), 'max_epochs': 2000,'metric': 'accuracy',
                   'n_internal_layers': (2, {'layer_1_2_nodes': 50.0, 'layer_2_2_nodes': 90.0, 'layer_3_2_nodes': 60.0}),
                   'optimizer': {'adam_params': None, 'name': 'Adam'}}
                  }
                  ],
                }



    testsets = [
                {'training': ['train_new_sample_1_2_greece_norm.csv'], \
                 'crossval': ['%s06*_attica_norm.csv' % tyear, '%s06*_sterea_norm.csv' % tyear]},
                {'training': ['train_new_sample_1_2_greece_norm.csv'], \
                 'crossval': ['%s07*_attica_norm.csv' % tyear, '%s07*_sterea_norm.csv' % tyear]},
                {'training': ['train_new_sample_1_2_greece_norm.csv'], \
                 'crossval': ['%s08*_attica_norm.csv' % tyear, '%s08*_sterea_norm.csv' % tyear]},
                {'training': ['train_new_sample_1_2_greece_norm.csv'], \
                 'crossval': ['%s09*_attica_norm.csv' % tyear, '%s09*_sterea_norm.csv' % tyear]},
        ]

    '''
    testsets = [
                {'training': ['train_new_sample_1_2_greece_norm.csv'], \
                 'crossval': ['%s06*_attica_norm.csv' % tyear]},
                {'training': ['train_new_sample_1_2_greece_norm.csv'], \
                 'crossval': ['%s07*_attica_norm.csv' % tyear]},
                {'training': ['train_new_sample_1_2_greece_norm.csv'], \
                 'crossval': ['%s08*_attica_norm.csv' % tyear]},
                {'training': ['train_new_sample_1_2_greece_norm.csv'], \
                 'crossval': ['%s09*_attica_norm.csv' % tyear]},
        ]
    '''

    calc_train_metrics = True
    opt_targets = ['BA', 'auc', 'f1-score 1', 'hybrid1', 'hybrid2', 'hybrid5', 'NH2', 'NH5', 'NH10']
    aucthress=2
    debug = True
    #modeltype = 'sk'
    modeltype = 'tf'
    class0_headrows = 0
    filespec = "ns_atster_do_%s"%tyear
    writescore = False
    resdir = '/mnt/nvme2tb/ffp/results/best2'
    cvrespattern = '*NN_gr*mean*'
    #cvrespattern=None
    #filters = ["df_flt['params'].str.contains(\"'dropout': None\")"] # no dropout
    filters = ["~df_flt['params'].str.contains(\"'dropout': None\")"] # with dropout
    #calib = {'min_temp':-0.15, 'dom_vel': -0.40, 'mean_temp': 0.2, 'mean_dew_temp': 0.2, 'min_dew_temp':0.2 , 'rain_7days': -0.999}
    iternum=5
    calib = {}
    #changeparams={'feature_drop': ('month', 'weekday', 'dom_dir', 'dir_max')+tuple(['corine_%d'%i for i in range(1,10)])}
    #changeparams = {'feature_drop': ('month', 'weekday', 'dom_dir', 'dir_max','pop','xpos','ypos') \
    #                    + tuple(['corine_%d' % i for i in range(1, 10)])}
    changeparams = None
    #xlaflags='--xla_gpu_cuda_data_dir=/usr/lib/cuda'
    nbest=3
    return testsets, space, testspace, cvrespattern, filters, nbest, changeparams, max_trials, calc_train_metrics, \
           opt_targets, trainsetdir, testsetdir, aucthress, modeltype, filespec, runmode, writescore, \
           resdir, iternum, calib, debug

