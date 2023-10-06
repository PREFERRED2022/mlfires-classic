import os

class space:

    def __init__(self, algo=None, recmetrics=None, writescore=None, region=None):
        self.space = None
        self.max_trials = 1
        self.trainsetdir = '/mnt/nvme2tb/ffp/datasets/train/'
        tyear = '2020'
        #self.testsetdir = os.path.join('/mnt/nvme2tb/ffp/datasets/test/',tyear)

        if region is not None:
            self.region = region
        else:
            self.region = ''

        self.testsetdir = os.path.join('/mnt/nvme2tb/ffp/datasets/test/',tyear,self.region)

        #runmode = 'val.'
        self.runmode = 'test'
        self.testspace=None

        #trainset = 'train_new_sample_1_2_norm.csv'
        trainset = 'train_new_sample_1_2_greece_norm.csv'

        self.testsets = [
                    {'training': [trainset], \
                     'crossval': ['%s06*df_%s_norm.csv' % (tyear, self.region)]},
                    {'training': [trainset], \
                     'crossval': ['%s07*df_%s_norm.csv' % (tyear, self.region)]},
                    {'training': [trainset], \
                     'crossval': ['%s08*df_%s_norm.csv' % (tyear, self.region)]},
                    {'training': [trainset], \
                     'crossval': ['%s09*df_%s_norm.csv' % (tyear, self.region)]},
            ]
        '''
        self.testsets = [
            {'training': [trainset], \
             'crossval': ['%s06*_attica_norm.csv' % tyear, '%s06*_sterea_norm.csv' % tyear]},
            {'training': [trainset], \
             'crossval': ['%s07*_attica_norm.csv' % tyear, '%s07*_sterea_norm.csv' % tyear]},
            {'training': [trainset], \
             'crossval': ['%s08*_attica_norm.csv' % tyear, '%s08*_sterea_norm.csv' % tyear]},
            {'training': [trainset], \
             'crossval': ['%s09*_attica_norm.csv' % tyear, '%s09*_sterea_norm.csv' % tyear]},
        ]
        
        self.testsets = [
            {'training': [trainset], \
             'crossval': ['%s06*_attica_norm.csv' % tyear]},
            {'training': [trainset], \
             'crossval': ['%s07*_attica_norm.csv' % tyear]},
            {'training': [trainset], \
             'crossval': ['%s08*_attica_norm.csv' % tyear]},
            {'training': [trainset], \
             'crossval': ['%s09*_attica_norm.csv' % tyear]},
        ]
        '''

        self.calc_test = True
        #recmetrics = ['BA', 'auc', 'f1-score 1', 'hybrid1', 'hybrid2', 'hybrid5', 'NH2', 'NH5', 'NH10']
        if recmetrics is not None:
            self.recmetrics = recmetrics
        else:
            self.recmetrics = ['NH5', 'NH10']
        #opt_targets = ['NH5']
        self.numaucthres=2
        self.debug = False
        self.modeltype = 'sk'
        #self.modeltype = 'tf'

        if self.modeltype == 'sk' and algo is not None:
            self.algo = algo
        elif self.modeltype == 'sk' and algo is None:
            self.algo = 'XGB'

        self.filespec = "ns_%s_%s_%s"%(self.region,tyear,self.algo)
        if writescore is not None:
            self.writescore = writescore
        else:
            self.writescore = False
        self.resdir = '/mnt/nvme2tb/ffp/results/best2'
        # self.testfpattern=None # when specific models are given
        self.testfpattern = '*%s*mean*'%self.algo # pattern of cross validation models csv files
        self.filters = []
        self.iternum=5
        self.calib = {}
        #self.changeparams={'feature_drop': ('month', 'weekday', 'dom_dir', 'dir_max')+tuple(['corine_%d'%i for i in range(1,10)])}
        #self.changeparams = {'feature_drop': ('month', 'weekday', 'dom_dir', 'dir_max','pop','xpos','ypos') \
        #                    + tuple(['corine_%d' % i for i in range(1, 10)])}
        self.changeparams = None
        self.nbest=3 # number of first best models to try

