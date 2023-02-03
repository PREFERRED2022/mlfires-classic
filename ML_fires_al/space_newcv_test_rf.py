from hyperopt import hp

def create_space():
    space = None
    max_trials = 1
    #trainsetdir = '/work2/pa21/sgirtsou/production/datasets/hard_cosine_similarity'
    trainsetdir = '/data2/ffp/datasets/trainingsets/randomnofire/'
    #testsetdir = '/work2/pa21/sgirtsou/production'
    #testsetdir = '/users/pa21/sgirtsou/production/2020'
    testsetdir = '/data2/ffp/datasets/daily/2020'
    resdir = '/data2/ffp/results/defCV'

    runmode ='test'
    
    nf = 90
  
    #best of def CV
    testspace = {  'NH5 %s'%runmode:[
                        {'params':{'algo': 'XT', 'n_estimators': 600, 'min_samples_split': 100, 'min_samples_leaf': 10, 'max_features': 53/nf*10, 
                         'max_depth': 22, 'criterion': 'gini', 'class_weights': {0: 1, 1: 10}, 'bootstrap': True, 'feature_drop': ['res_max']}},
                        {'params':{'algo': 'RF', 'n_estimators': 250, 'min_samples_split': 180, 'min_samples_leaf': 40, 'max_features': 41/nf*10, 'max_depth': 20, 
                         'criterion': 'entropy', 'class_weights': {0: 1, 1: 9}, 'bootstrap': True, 'feature_drop': ['res_max']}},
                        {'params':{'algo': 'XGB', 'subsample': 0.5, 'scale_pos_weight': 1000, 'n_estimators': 800, 'max_depth': 4, 'lambda': 17, 'gamma': 10, 
                         'alpha': 100, 'feature_drop': ['res_max']}}
                        ],
                   'NH10 %s'%runmode:[
                        {'params':{'algo': 'XGB', 'subsample': 0.9, 'scale_pos_weight': 40000, 'n_estimators': 200, 'max_depth': 30, 'lambda': 18, 'gamma': 100, 
                         'alpha': 80, 'feature_drop': ['res_max']}},
                        ]
                }
    '''
    #best of AltCv
    testspace = { 'hybrid5 %s'%runmode:[
                        {'params':{'algo': 'XT', 'bootstrap': False, 'class_weights': {0: 1, 1: 10}, 'criterion': 'entropy', 'feature_drop': ['res_max'],
                         'max_depth': 18.0, 'max_features': 7.0, 'min_samples_leaf': 25, 'min_samples_split': 10, 'n_estimators': 800}},
                  ],
                  'NH2 %s'%runmode:[
                        {'params':{'algo': 'XT', 'bootstrap': False, 'class_weights': {0: 1, 1: 50}, 'criterion': 'entropy', 'feature_drop': ['res_max'], 'max_depth': 20.0, 
                         'max_features': 9.0, 'min_samples_leaf': 20, 'min_samples_split': 2, 'n_estimators': 1000}},
                        {'params':{'algo': 'XGB', 'alpha': 80, 'feature_drop': ['res_max'], 'gamma': 10, 'lambda': 21.0, 'max_depth': 44.0, 'n_estimators': 40, 
                         'scale_pos_weight': 70, 'subsample': 0.7}}
                  ],
                  'NH5 %s'%runmode:[
                        {'params':{'algo': 'XT', 'bootstrap': False, 'class_weights': {0: 1, 1: 50}, 'criterion': 'entropy', 'feature_drop': ['res_max'], 
                        'max_depth': 20.0, 'max_features': 9.0, 'min_samples_leaf': 20, 'min_samples_split': 2, 'n_estimators': 1000}}
                  ]
                }
    
    # res_max test
    testspace = { 'hybrid5 %s'%runmode:[
                        {'params':{'algo': 'XT', 'bootstrap': False, 'class_weights': {0: 1, 1: 10}, 'criterion': 'entropy', 'feature_drop': ['res_max'],
                         'max_depth': 18.0, 'max_features': 7.0, 'min_samples_leaf': 25, 'min_samples_split': 10, 'n_estimators': 800}},
                        {'params':{'algo': 'XT', 'bootstrap': False, 'class_weights': {0: 1, 1: 10}, 'criterion': 'entropy', 'feature_drop': [],
                         'max_depth': 18.0, 'max_features': 7.0, 'min_samples_leaf': 25, 'min_samples_split': 10, 'n_estimators': 800}},
                        {'params':{'algo': 'RF', 'bootstrap': True, 'class_weights': {0: 1, 1: 300}, 'criterion': 'entropy', 'feature_drop': [],
                         'max_depth': 1200, 'max_features': 1.0, 'min_samples_leaf': 10, 'min_samples_split': 2, 'n_estimators': 750}},
                        {'params':{'algo': 'RF', 'bootstrap': True, 'class_weights': {0: 1, 1: 300}, 'criterion': 'entropy', 'feature_drop': ['res_max'],
                         'max_depth': 1200, 'max_features': 1.0, 'min_samples_leaf': 10, 'min_samples_split': 2, 'n_estimators': 750}},
                        {'params':{'algo': 'XGB', 'alpha': 80, 'feature_drop': [], 'gamma': 10, 'lambda': 3.0, 'max_depth': 66.0, 'n_estimators': 60, 'scale_pos_weight': 50, 'subsample': 0.5}},
                        {'params':{'algo': 'XGB', 'alpha': 80, 'feature_drop': ['res_max'], 'gamma': 10, 'lambda': 3.0, 'max_depth': 66.0, 'n_estimators': 60, 'scale_pos_weight': 50, 'subsample': 0.5}},
                        {'params':{'algo': 'XT', 'bootstrap': False, 'class_weights': {0: 1, 1: 10}, 'criterion': 'entropy', 'feature_drop': ['res_max'],
                         'max_depth': 18.0, 'max_features': 7.0, 'min_samples_leaf': 25, 'min_samples_split': 10, 'n_estimators': 800}},
                        {'params':{'algo': 'XT', 'bootstrap': False, 'class_weights': {0: 1, 1: 10}, 'criterion': 'entropy', 'feature_drop': [],
                         'max_depth': 18.0, 'max_features': 7.0, 'min_samples_leaf': 25, 'min_samples_split': 10, 'n_estimators': 800}},
                        {'params':{'algo': 'RF', 'bootstrap': True, 'class_weights': {0: 1, 1: 300}, 'criterion': 'entropy', 'feature_drop': [],
                         'max_depth': 1200, 'max_features': 1.0, 'min_samples_leaf': 10, 'min_samples_split': 2, 'n_estimators': 750}},
                        {'params':{'algo': 'RF', 'bootstrap': True, 'class_weights': {0: 1, 1: 300}, 'criterion': 'entropy', 'feature_drop': ['res_max'],
                         'max_depth': 1200, 'max_features': 1.0, 'min_samples_leaf': 10, 'min_samples_split': 2, 'n_estimators': 750}},
                        {'params':{'algo': 'XGB', 'alpha': 80, 'feature_drop': [], 'gamma': 10, 'lambda': 3.0, 'max_depth': 66.0, 'n_estimators': 60, 'scale_pos_weight': 50, 'subsample': 0.5}},
                        {'params':{'algo': 'XGB', 'alpha': 80, 'feature_drop': ['res_max'], 'gamma': 10, 'lambda': 3.0, 'max_depth': 66.0, 'n_estimators': 60, 'scale_pos_weight': 50, 'subsample': 0.5}}
                     ],
                }

    
    testspace = { 'hybrid5 %s'%runmode:[
                        {'params':{'algo': 'XT', 'bootstrap': False, 'class_weights': {0: 1, 1: 10}, 'criterion': 'entropy', 'feature_drop': ['res_max'],
                         'max_depth': 18.0, 'max_features': 7.0, 'min_samples_leaf': 25, 'min_samples_split': 10, 'n_estimators': 800}},
                        ],
                }
    '''

    #testspace = None
    tyear = '2020'
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
                #{'training':['old_random_new_feat_from_months.csv'],\
                # 'crossval':['*_%s_norm.csv'%tyear]}
                #{'training':['old_random_new_feat_from_months.csv'],\
                # 'crossval':['202108*_norm.csv']}
                {'training':['old_random_new_feat_from_months.csv'],\
                 'crossval':['%s06*_norm.csv'%tyear]},
                {'training':['old_random_new_feat_from_months.csv'],\
                 'crossval':['%s07*_norm.csv'%tyear]},
                {'training':['old_random_new_feat_from_months.csv'],\
                 'crossval':['%s08*_norm.csv'%tyear]},
                {'training':['old_random_new_feat_from_months.csv'],\
                 'crossval':['%s09*_norm.csv'%tyear]},
               ]

    calc_train_metrics = True
    #opt_targets = ['auc', 'f1-score 1', 'hybrid2', 'hybrid5', 'NH2', 'NH5', 'NH10']
    opt_targets = ['f1-score 1']
    debug = True
    modeltype = 'sk'
    #modeltype = 'tf'
    cvrownum = 0
    filedesc = "2020_fix_defCV"
    aucthress = 0
    writescores=True
    #cvrespattern = '*XT_1M_sb*'
    #cvrespattern = '*XT_1M*mean*'
    cvrespattern=None
    return testsets, space, testspace, cvrespattern, [], max_trials, 'random', calc_train_metrics, opt_targets, trainsetdir, testsetdir,\
           aucthress, modeltype, cvrownum, filedesc, runmode, writescores, resdir, debug

