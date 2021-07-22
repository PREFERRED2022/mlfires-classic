#!/usr/bin/env python
# coding: utf-8
import json

import pandas as pd
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
import tensorflow.keras.metrics
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import clone_model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from sklearn.feature_selection import SequentialFeatureSelector
from functools import partial
from multiprocessing import Pool
from itertools import combinations
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split

def create_NN_model(params, X):
    # define model
    model = Sequential()
    n_features = X.shape[1]
    intlayers = int(params['n_internal_layers'][0])
    model.add(Dense(params['n_internal_layers'][1]['layer_1_' + str(intlayers) + '_nodes'], activation='relu',))# input_shape=(n_features,)))
    if not params['dropout'] is None:
        model.add(Dropout(params['dropout']))
    for i in range(2, intlayers + 2):
        model.add(Dense(int(params['n_internal_layers'][1]['layer_' + str(i) + '_' + str(intlayers) + '_nodes']),
                        activation='relu', )) #kernel_initializer=initializer))
        if not params['dropout'] is None:
            model.add(Dropout(params['dropout']))
    model.add(Dense(2, activation='softmax'))

    # compile the model
    if params['optimizer']['name']=='Adam':
        if params['optimizer']['adam_params'] is None:
            opt = Adam()
        else:
            opt = Adam(learning_rate=params['optimizer']['adam_params']['learning_rate_adam'], beta_1=params['optimizer']['adam_params']['beta_1'],
                       beta_2=params['optimizer']['adam_params']['beta_2'],amsgrad=params['optimizer']['adam_params']['amsgrad'])
    elif params['optimizer']['name']=='SGD':
        opt = SGD(learning_rate=params['optimizer']['learning_rate_SGD'])

    if params['metric'] == 'accuracy':
        metrics = ['accuracy']
    elif params['metric'] == 'sparse':
        metrics = [tensorflow.metrics.SparseCategoricalAccuracy()]
    elif params['metric'] == 'tn':
        metrics = [tensorflow.metrics.TrueNegatives(),tensorflow.metrics.TruePositives()]
    if 'loss' in params and params['loss'] == 'unbalanced':
        lossf='sparse_categorical_crossentropy'
    else:
        lossf='sparse_categorical_crossentropy'
    model.compile(optimizer=opt, loss=lossf, metrics=metrics)  # , AUC(multi_label=False)])
    return model


def load_dataset(trfiles, featuredrop=[], debug=True, returnid=False):
    # dsfile = 'dataset_ndvi_lu.csv'
    domdircheck = 'dom_dir'
    dirmaxcheck = 'dir_max'
    corinecheck = 'Corine'
    monthcheck = 'month'
    wkdcheck = 'wkd'
    firedatecheck = 'firedate'
    X_columns = ['max_temp', 'min_temp', 'mean_temp', 'res_max', dirmaxcheck, 'dom_vel', domdircheck,
                 'rain_7days', corinecheck, 'Slope', 'DEM', 'Curvature', 'Aspect', 'ndvi', 'evi', 'lst_day',
                 'lst_night', monthcheck, wkdcheck,
                 'mean_dew_temp', 'max_dew_temp', 'min_dew_temp','frequency', 'f81', 'x', 'y']
    y_columns = ['fire']
    # if not os.path.exists(os.path.join(dsetfolder, dsready)):
    if isinstance(trfiles, list):
        if debug:
            print("Loading full dataset ...")
        dflist=[]
        for dsfile in trfiles:
            if debug:
                print("Loading dataset file %s" % dsfile)
            dflist.append(pd.read_csv(dsfile))
        df = pd.concat(dflist)
    else:
        dsfile = trfiles
    df = pd.read_csv(dsfile)
    X_columns_upper = [c.upper() for c in X_columns]
    newcols = [c for c in df.columns if
               c.upper() in X_columns_upper or any([cX in c.upper() for cX in X_columns_upper])]
    X_columns = newcols
    #corine_col, newcols = check_categorical(df, corinecheck, newcols)
    #dirmax_col, newcols = check_categorical(df, dirmaxcheck, newcols)
    #domdir_col, newcols = check_categorical(df, domdircheck, newcols)
    #month_col, newcols = check_categorical(df, monthcheck, newcols)
    #wkd_col, newcols = check_categorical(df, wkdcheck, newcols)

    firedate_col = [c for c in df.columns if firedatecheck.upper() in c.upper()][0]
    X, y, groupspd = prepare_dataset(df, X_columns, y_columns, firedate_col)
    print("Ignored columns from csv %s"%([c for c in df.columns if c not in X.columns]))
    idpd = df['id']
    df = None
    X_columns = X.columns
    if len(featuredrop) > 0:
        X = X.drop(columns=[c for c in X.columns if any([fd in c for fd in featuredrop])])
    print("Dropped columns %s"%(list(set(X_columns)-set(X.columns))))
    #if debug:
    #    print("X helth check %s"%X.describe())
    #    print("y helth check %s"%y.describe())
    if returnid:
        return X, y, groupspd, idpd
    else:
        return X, y, groupspd

def prepare_dataset(df, X_columns, y_columns, firedate_col):
    df = df[X_columns+y_columns+[firedate_col]]
    print('before nan drop: %d' % len(df.index))
    df = df.dropna()
    print('after nan drop: %d' % len(df.index))
    df = df.drop_duplicates(keep='first')
    df.reset_index(inplace=True, drop=True)
    print('after dup. drop: %d' % len(df.index))
    print('renaming "x": "xpos", "y": "ypos"')
    X_unnorm, y_int = df[X_columns], df[y_columns]
    X_unnorm = X_unnorm.rename(columns={'x': 'xpos', 'y': 'ypos'})
    # X = normdataset.normalize_dataset(X_unnorm, aggrfile='stats/featurestats.json')
    X = X_unnorm
    y = y_int
    groupspd = df[firedate_col]
    return X, y, groupspd

class MakeModel(object):

    def __init__(self, X=None, y=None):
        pass

    def predict(self, X):
        y_pred = self.model.predict(X)
        return y_pred
        #return np.argmax(y_pred, axis=1)
    
    def fit(self, X, y, epochs=250):
        skwrapped_model = KerasClassifier(build_fn=creatennmodel,
                                          #train_input=X,
                                          epochs=epochs,
                                          batch_size=512,
                                          #validation_split=1-TRAIN_TEST_SPLIT,
                                          verbose=0)
        self.model = skwrapped_model
        self.model.fit(X, y)
        return self.model

def sfssknn(sknnmodel, featnum):
    sfs = SequentialFeatureSelector(sknnmodel, n_features_to_select=featnum, direction='backward')
    sfs.fit(X, y)
    bestfeatmask=sfs.get_support()
    with open('results/sknnSFS_feat_%d'%featnum,"a") as f:
        for m in [True,False]:
            f.write('included features : %s\n' % m)
            fn = 0
            for i in range(0, len(bestfeatmask)):
                if bestfeatmask[i]==m:
                    fn += 1
                    f.write('feature %d : %s\n' % (fn, X.columns[i]))
            f.write('\n')

with open('featrankconf.json','r') as fconf:
    confst = fconf.read()
    configuration = json.loads(confst)

params=eval(configuration['params'])
featdrop = [] if not eval(configuration['drop']) else params['feature_drop']
X, y, g=load_dataset('/home/aapostolakis/Documents/ffpdata/newcrossval/datasets/randomnofire/oldrandomnewfeat.csv', featuredrop=featdrop)
creatennmodel = partial(create_NN_model, params, X)
sknnmodel = KerasClassifier(build_fn=creatennmodel, batch_size=params['batch_size'], epochs=configuration['epochs'], verbose=0,)

sfssknn(sknnmodel, 89)
'''
sfssknnp = partial(sfssknn, sknnmodel)
with Pool(8) as p:
     p.map(sfssknn, list(range(91,70-2)))
'''

