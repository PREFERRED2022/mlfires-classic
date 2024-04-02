import os

def create_space():
    space = None
    max_trials = 1
    trainsetdir = '/mnt/nvme2tb/ffp/datasets/train/'
    tyear = '2020'
    testsetdir = os.path.join('/mnt/nvme2tb/ffp/datasets/test/',tyear,'greece')
    #runmode = 'val.'
    runmode = 'test'
    testspace=None
    '''  
    testsets = [
                {'training': ['train_new_sample_1_2_greece_norm.csv'], \
                 'crossval': ['%s06*df_greece_norm.csv' % tyear]},
                {'training': ['train_new_sample_1_2_greece_norm.csv'], \
                 'crossval': ['%s07*df_greece_norm.csv' % tyear]},
                {'training': ['train_new_sample_1_2_greece_norm.csv'], \
                 'crossval': ['%s08*df_greece_norm.csv' % tyear]},
                {'training': ['train_new_sample_1_2_greece_norm.csv'], \
                 'crossval': ['%s09*df_greece_norm.csv' % tyear]},
        ]
    '''
    testsets = [
                {'training': ['train_new_sample_1_2_norm.csv'], \
                 'crossval': ['%s06*df_norm.csv' % tyear]},
                {'training': ['train_new_sample_1_2_norm.csv'], \
                 'crossval': ['%s07*df_norm.csv' % tyear]},
                {'training': ['train_new_sample_1_2_norm.csv'], \
                 'crossval': ['%s08*df_norm.csv' % tyear]},
                {'training': ['train_new_sample_1_2_norm.csv'], \
                 'crossval': ['%s09*df_norm.csv' % tyear]},
        ]

    calc_train_metrics = True
    #opt_targets = ['BA', 'auc', 'f1-score 1', 'hybrid1', 'hybrid2', 'hybrid5', 'NH2', 'NH5', 'NH10']
    #opt_targets = ['hybrid1', 'hybrid2', 'hybrid5', 'NH2', 'NH5', 'NH10']
    opt_targets = ['NH5', 'NH10']
    #opt_targets = ['NH5']
    aucthress=2
    debug = True
    #modeltype = 'sk'
    modeltype = 'tf'
    class0_headrows = 0
    filespec = "ns_gr_%s"%tyear
    writescore = False
    resdir = '/mnt/nvme2tb/ffp/results/bestmodels'
    cvrespattern = '*NN_ns*mean*' # pattern of cross validation models csv
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

