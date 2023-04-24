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

    nf = 90

    # best of def CV
    '''
    testspace = {'NH5 %s' % runmode: [
        {'params': {'algo': 'XT', 'n_estimators': 600, 'min_samples_split': 100, 'min_samples_leaf': 10,
                    'max_features': 53 / nf * 10,
                    'max_depth': 22, 'criterion': 'gini', 'class_weights': {0: 1, 1: 10}, 'bootstrap': True,
                    'feature_drop': ['res_max']}},
        {'params': {'algo': 'RF', 'n_estimators': 250, 'min_samples_split': 180, 'min_samples_leaf': 40,
                    'max_features': 41 / nf * 10, 'max_depth': 20,
                    'criterion': 'entropy', 'class_weights': {0: 1, 1: 9}, 'bootstrap': True,
                    'feature_drop': ['res_max']}},
        {'params': {'algo': 'XGB', 'subsample': 0.5, 'scale_pos_weight': 1000, 'n_estimators': 800, 'max_depth': 4,
                    'lambda': 17, 'gamma': 10,
                    'alpha': 100, 'feature_drop': ['res_max']}}
    ],
        'NH10 %s' % runmode: [
            {'params': {'algo': 'XGB', 'subsample': 0.9, 'scale_pos_weight': 40000, 'n_estimators': 200,
                        'max_depth': 30, 'lambda': 18, 'gamma': 100,
                        'alpha': 80, 'feature_drop': ['res_max']}},
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
    modeltype = 'sk'
    #modeltype = 'tf'
    class0_headrows = 0
    filespec = "%s_XT"%tyear
    writescore = False
    resdir = '/mnt/nvme2tb/ffp/results/newdefCV/'
    #resdir = '/work2/pa21/sgirtsou/production/results/newcv/nn/'
    #cvrespattern = '*_dropfeat_*_mean*'
    #cvrespattern = '*_dropfeat_1M_*_mean*'
    #cvrespattern = '*RF*mean*'
    cvrespattern = '*XT*mean*'
    #cvrespattern = '*XT*mean*'
    #filters = ["df_flt['params'].str.contains(\"'dropout': None\")"]
    #filters = ["~df_flt['params'].str.contains(\"'dropout': None\")"]
    filters=[]
    #calib = {'min_temp':-0.15, 'dom_vel': -0.40, 'mean_temp': 0.2, 'mean_dew_temp': 0.2, 'min_dew_temp':0.2 , 'rain_7days': -0.999}
    calib = {}
    #xlaflags='--xla_gpu_cuda_data_dir=/usr/lib/cuda'
    return testsets, space, None, cvrespattern, filters, max_trials, calc_train_metrics, opt_targets, trainsetdir, testsetdir,\
           aucthress, modeltype, filespec, runmode, writescore, resdir, calib, debug

