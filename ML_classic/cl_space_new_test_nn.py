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

        self.testsets = [
                    {'training': ['train_new_sample_1_2_greece_norm.csv'], \
                     'crossval': ['%s06*df_%s_norm.csv' % (tyear, self.region)]},
                    {'training': ['train_new_sample_1_2_greece_norm.csv'], \
                     'crossval': ['%s07*df_%s_norm.csv' % (tyear, self.region)]},
                    {'training': ['train_new_sample_1_2_greece_norm.csv'], \
                     'crossval': ['%s08*df_%s_norm.csv' % (tyear, self.region)]},
                    {'training': ['train_new_sample_1_2_greece_norm.csv'], \
                     'crossval': ['%s09*df_%s_norm.csv' % (tyear, self.region)]},
            ]
        '''
        self.testsets = [
                    {'training': ['train_new_sample_1_2_norm.csv'], \
                     'crossval': ['%s06*df_norm.csv' % tyear]},
                    {'training': ['train_new_sample_1_2_norm.csv'], \
                     'crossval': ['%s07*df_norm.csv' % tyear]},
                    {'training': ['train_new_sample_1_2_norm.csv'], \
                     'crossval': ['%s08*df_norm.csv' % tyear]},
                    {'training': ['train_new_sample_1_2_norm.csv'], \
                     'crossval': ['%s09*df_norm.csv' % tyear]},
            ]
        '''
        self.calc_test = True
        #opt_targets = ['BA', 'auc', 'f1-score 1', 'hybrid1', 'hybrid2', 'hybrid5', 'NH2', 'NH5', 'NH10']
        if recmetrics is not None:
            self.recmetrics = recmetrics
        else:
            self.recmetrics = ['NH5', 'NH10']
        #opt_targets = ['NH5']
        self.numaucthres=2
        self.debug = False
        #modeltype = 'sk'
        self.modeltype = 'tf'

        if self.modeltype == 'tf' and algo is not None:
            if algo == 'do':
                self.filters = ["~df_flt['params'].str.contains(\"'dropout': None\")"]  # with dropout
            elif algo == 'ndo':
                self.filters = ["df_flt['params'].str.contains(\"'dropout': None\")"]  # no dropout
        elif self.modeltype == 'tf' and algo is None:
            self.filters = ["~df_flt['params'].str.contains(\"'dropout': None\")"]  # with dropout
            #self.filters = ["df_flt['params'].str.contains(\"'dropout': None\")"]  # no dropout

        self.filespec = "ns_%s_%s"%(self.region,tyear)
        if writescore is not None:
            self.writescore = writescore
        else:
            self.writescore = False
        self.resdir = '/mnt/nvme2tb/ffp/results/bestmodels'
        self.testfpattern = '*NN_ns*mean*' # pattern of cross validation models csv files
        #calib = {'min_temp':-0.15, 'dom_vel': -0.40, 'mean_temp': 0.2, 'mean_dew_temp': 0.2, 'min_dew_temp':0.2 , 'rain_7days': -0.999}
        self.iternum=5
        self.calib = {}
        #self.modelfile=None
        self.modelfile= "/mnt/nvme2tb/ffp/results/bestmodels/hypres_tf_ns_ncv_do_2019_model_id_785_r_0_hybrid2test_1.h5"
        #changeparams={'feature_drop': ('month', 'weekday', 'dom_dir', 'dir_max')+tuple(['corine_%d'%i for i in range(1,10)])}
        #changeparams = {'feature_drop': ('month', 'weekday', 'dom_dir', 'dir_max','pop','xpos','ypos') \
        #                    + tuple(['corine_%d' % i for i in range(1, 10)])}
        self.changeparams = None
        self.nbest=3


