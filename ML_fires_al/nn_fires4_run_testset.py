#!/usr/bin/env python
from hyperopt import Trials, fmin, tpe, hp, STATUS_OK
from pandas import read_csv
from sklearn.model_selection import train_test_split, KFold, GroupKFold
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import models
import tensorflow.keras.metrics
from tensorflow.keras.utils import to_categorical
import numpy as np
from pandas import concat
from pandas import DataFrame
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd
import normdataset
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow import summary
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
import time
import json
import space_test
import re
from datetime import datetime

def drop_all0_features(df):
    for c in df.columns:
        if 'bin' in c:
            u = df[c].unique()
            if (len(u)>1):
                cnt = df.groupby([c]).count()
                if cnt["max_temp"][1]<20:
                    print("Droping Feature %s, count: %d"%(c, cnt["max_temp"][1]))
                    df.drop(columns=[c])
            else:
                print("%s not exists (size : %d)"%(c, len(u)))

def prepare_dataset(df, X_columns, y_columns, firedate_col, corine_col, domdir_col, dirmax_col, statfile):
    # df = read_csv('/home/sgirtsou/Documents/ML-dataset_newLU/training_dataset.csv')
    #df = df.dropna()
    #df = df[~df.isin(['--']).any(axis=1)]
    #df = df[(df != -1000).any(axis=1)]

    #df.columns = ['id', 'firedate_x', 'max_temp', 'min_temp', 'mean_temp', 'res_max',
    #              'dir_max', 'dom_vel', 'dom_dir', 'rain_7days', 'Corine', 'Forest',
    #              'fire', 'firedate_g', 'firedate_y', 'tile', 'max_temp_y', 'DEM',
    #              'Slope', 'Curvature', 'Aspect', 'image', 'ndvi']
    #df_part = df[
    #    ['id', 'max_temp', 'min_temp', 'mean_temp', 'res_max', 'dir_max', 'dom_vel', 'dom_dir', 'rain_7days', 'Corine',
    #     'Slope', 'DEM', 'Curvature', 'Aspect', 'ndvi', 'fire']]

    X_unnorm, y_int = df[X_columns], df[y_columns]

    # categories to binary
    if domdir_col:
        Xbindomdir = pd.get_dummies(X_unnorm[domdir_col].round())
        if 0 in Xbindomdir.columns:
            del Xbindomdir[0]
        ddircols = []
        for i in range(1, 9):
            ddircols.append('bin_dom_dir_%d' % i)
        Xbindomdir.columns = ddircols
        del X_unnorm[domdir_col]
        X_unnorm = pd.concat([X_unnorm, Xbindomdir], axis = 1)

    if dirmax_col:
        Xbindirmax = pd.get_dummies(X_unnorm[dirmax_col].round())
        if 0 in Xbindirmax.columns:
            del Xbindirmax[0]
        dmaxcols = []
        for i in range(1, 9):
            dmaxcols.append('bin_dir_max_%d' % i)
        Xbindirmax.columns = dmaxcols
        del X_unnorm[dirmax_col]
        X_unnorm = pd.concat([X_unnorm, Xbindirmax], axis = 1)

    if corine_col:
        # convert corine level
        corine2 = X_unnorm[corine_col].copy().astype('int32') // 10
        del X_unnorm[corine_col]
        #X_unnorm.rename(columns={corine_col: 'corine_orig'})
        X_unnorm = pd.concat([X_unnorm, corine2], axis=1)

        Xbincorine = pd.get_dummies(X_unnorm[corine_col])
        corcols = ['bin_corine_' + str(c) for c in Xbincorine.columns]
        Xbincorine.columns = corcols
        del X_unnorm[corine_col]
        X_unnorm = pd.concat([X_unnorm, Xbincorine], axis = 1)

    #str_classes = ['Corine']
    #X_unnorm_int = normdataset.index_string_values(X_unnorm, str_classes)
    #X = normdataset.normalize_dataset(X_unnorm_int, 'std')

    X = normdataset.normalize_dataset(X_unnorm, aggrfile = statfile)
    y = y_int
    if 'id' in df.columns:
        groupspd = df[['id', firedate_col]]
    else:
        groupspd = df[[firedate_col]]

    return X, y, groupspd

def check_categorical(df, checkcol, newcols):
    cat_cols = [c for c in df.columns if checkcol.upper() in c.upper()]
    if any([c.upper() == checkcol.upper() for c in cat_cols]) and len(cat_cols) > 1:
        cat_col = [c for c in df.columns if checkcol.upper() == c.upper()][0]
        deletecolumns = []
        for c in newcols:
            if (c.upper() != checkcol.upper() and checkcol.upper() in c.upper()):
                deletecolumns.append(c)
        for c in deletecolumns:
            newcols.remove(c)
    elif any([c.upper() == checkcol.upper() for c in cat_cols]) and len(cat_cols) == 1:
        cat_col = [c for c in df.columns if checkcol.upper() == c.upper()][0]
    elif not any([c.upper() == checkcol.upper() for c in cat_cols]) and len(cat_cols) == 1:
        cat_col = [c for c in df.columns if checkcol.upper() == c.upper()][0]
    elif not any([c.upper() == checkcol.upper() for c in cat_cols]) and len(cat_cols) > 1:
        cat_col = None
        for i in range(0,len(newcols)):
            c = newcols[i]
            if (c.upper() != checkcol.upper() and checkcol.upper() in c.upper()):
                newname = "bin_" + c
                newcols[i] = newname
                df.rename(columns={c : newname}, inplace=True)
    else:
        cat_col = None
    return cat_col, newcols

def fix_categorical(df, colchecklist, Xcols, drop_columns):
    X_columns_upper = [c.upper() for c in Xcols]
    newcols = [c for c in df.columns if c.upper() in X_columns_upper or any([cX in c.upper() for cX in X_columns_upper])]
    colsbasenames = []
    for c in colchecklist:
        colbasename, newcols = check_categorical(df, c, newcols)
        colsbasenames.append(colbasename)
    newcols = [c for c in newcols if not any([dc in c for dc in drop_columns])]
    return newcols, colsbasenames[0], colsbasenames[1], colsbasenames[2]

def updatestats(df, prevstats, newcols):
    for c in newcols:
        if not c in prevstats:
            prevstats[c] = {}
        prevstats[c]['max'] = max(df[c].max(),prevstats[c]['max']) if 'max' in prevstats[c] else df[c].max()
        prevstats[c]['min'] = min(df[c].min(),prevstats[c]['min']) if 'min' in prevstats[c] else df[c].min()

def npconv(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime):
        return obj.__str__()

def create_unnorm_datefile(dsetfolder, dfchunk, cols, y_columns, dsready, dsunnormsuffix, setdate):
    fileunnorm = os.path.join(dsetfolder, "%s%s_%s%s" % (os.path.basename(dsready), dsunnormsuffix, setdate, ".csv"))
    if not os.path.exists(fileunnorm):
        pd.concat([dfchunk[cols], dfchunk[y_columns]], axis=1).to_csv(fileunnorm, index=False)

def filenames(dsfile):
    dsetfolder = 'data/'
    dsreadysuffix = '_ready'
    dsunnormsuffix = '_unnorm'
    dsready = dsfile[:-4] + dsreadysuffix
    dsfile = os.path.join(dsetfolder, dsfile) if not os.path.dirname(dsfile) else dsfile
    dsetfolder = os.path.join(os.path.dirname(dsfile), 'ready')
    return dsetfolder, dsreadysuffix, dsunnormsuffix, dsfile, dsetfolder, dsready

# load the dataset
def load_datasets(dsfile, perdate = False, calcstats = None, statfname = None, checkunnorm = True):
    dsetfolder, dsreadysuffix, dsunnormsuffix, dsfile, dsetfolder, dsready = filenames(dsfile)
    domdircheck = 'dom_dir'
    dirmaxcheck = 'dir_max'
    corinecheck = 'Corine'
    firedatecheck = 'firedate'
    X_columns = ['max_temp', 'min_temp', 'mean_temp', 'res_max', dirmaxcheck, 'dom_vel', domdircheck,
                 'rain_7days',
                 corinecheck, 'Slope', 'DEM', 'Curvature', 'Aspect', 'ndvi_new', 'evi']
    #X_columns = ['ndvi', 'max_temp', 'min_temp', 'mean_temp', 'res_max', dirmaxcheck, 'dom_vel', domdircheck,
    #             'rain_7days',
    #             corinecheck, 'Slope', 'DEM', 'Curvature', 'Aspect', 'evi']
    drop_columns = ['-1000']
    y_columns = ['fire']
    chunksize = 3.75 * 10 ** 5  # about 1 day
    cnt = 0
    df1 = pd.read_csv(dsfile, nrows=1 )
    firedate_col = [c for c in df1.columns if firedatecheck.upper() in c.upper()][0]
    if checkunnorm:
        for dfchunk in pd.read_csv(dsfile, chunksize=chunksize, dtype ={ firedate_col: 'str'}):
            if 'id' in dfchunk.columns:
                dfchunk = dfchunk.astype({'id': 'int32'})
            firstdate = dfchunk[firedate_col].head(1).item() if perdate else ""
            print("df first date: %s shape: %s" % (firstdate, dfchunk.shape))
            newcols, corine_col, dirmax_col, domdir_col = fix_categorical(dfchunk, [corinecheck, dirmaxcheck, domdircheck], X_columns, drop_columns)
            if os.path.exists(os.path.join(dsetfolder, "%s_%s%s"%(os.path.basename(dsready),firstdate,".csv"))):
                print('Date: %s normalized exists' % firstdate)
                dfrest = dfchunk.loc[dfchunk[firedate_col] != firstdate]
                cnt += 1
                continue
            if cnt == 0 and perdate:
                dfrest = dfchunk.loc[dfchunk[firedate_col] != firstdate]
                dfchunk = dfchunk.loc[dfchunk[firedate_col] == firstdate]
            elif perdate and dfrest.size > 0:
                alldates = dfrest[firedate_col].unique()
                if alldates.size>1:
                    print('more than one dates %s'%alldates)
                firstdate = alldates[alldates.size-1]
                #firstdate = dfrest[firedate_col].head(1).item()
                dftemp = pd.concat([dfrest.loc[dfrest[firedate_col] == firstdate], dfchunk.loc[dfchunk[firedate_col] == firstdate]])
                dfrest = dfchunk.loc[dfchunk[firedate_col] != firstdate]
                dfchunk = dftemp
            elif not perdate:
                firstdate=""
            print('Date: %s'%firstdate)

            cnt += 1

            dfchunk = dfchunk.dropna()
            print("df date: %s shape: %s" % (firstdate, dfchunk.shape))
            dfchunk = dfchunk[(dfchunk != '--').all(axis=1)]
            print("drop -- : %s"%(dfchunk.shape[0]))
            dfchunk = dfchunk[(dfchunk != -1000).all(axis=1)]
            print("drop -1000 : %s"%(dfchunk.shape[0]))
            create_unnorm_datefile(dsetfolder, dfchunk, [firedate_col]+['id']+newcols, y_columns, dsready, dsunnormsuffix, firstdate)
            if not calcstats is None:
                updatestats(dfchunk, calcstats, newcols)
            else:
                X, y, groupspd = prepare_dataset(dfchunk, newcols, y_columns, firedate_col, corine_col, domdir_col, dirmax_col, statfname)
                featdf = pd.concat([groupspd, X, y], axis=1)
                featdf = featdf[[c for c in featdf.columns if 'Unnamed' not in c]]
                featdf.to_csv(os.path.join(dsetfolder, "%s_%s%s"%(os.path.basename(dsready),firstdate,".csv")), index=False)
        if calcstats:
            return calcstats
    flist = [fn for fn in os.listdir(dsetfolder) if os.path.basename(dsready) in fn and dsunnormsuffix not in fn and "~" not in fn]
    for dsready in flist:
        print("Loading %s"%os.path.join(dsetfolder, dsready))
        featdf = pd.read_csv(os.path.join(dsetfolder, dsready), index_col=False)
        firedate_col = [c for c in featdf.columns if firedate_col.upper() in c.upper()][0]
        firstdate = featdf[firedate_col].head(1).item()
        X_columns_new = [c for c in featdf.columns if c not in [firedate_col]+y_columns+['id']+[c for c in featdf.columns if 'scores' in c] and 'Unnamed' not in c]
        #X_columns_new = [c for c in featdf.columns if c not in y_columns and 'Unnamed' not in c]
        X = featdf[X_columns_new]
        y = featdf[y_columns]
        #groupspd = featdf[firedate_col]
        yield X, y, firstdate

    #return X, y, groupspd


def create_NN_model(params, X):
    # define model
    model = Sequential()
    n_features = X.shape[1]
    intlayers = int(params['n_internal_layers'][0])
    model.add(Dense(params['n_internal_layers'][1]['layer_1_' + str(intlayers) + '_nodes'], activation='relu',
                    input_shape=(n_features,)))
    for i in range(2, intlayers + 2):
        model.add(Dense(int(params['n_internal_layers'][1]['layer_' + str(i) + '_' + str(intlayers) + '_nodes']),
                        activation='relu'))
        # model.add(Dense(1, activation='sigmoid'))
    model.add(Dense(2, activation='softmax'))

    # compile the model

    from tensorflow.keras.optimizers import Adam

    adam = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)

    if params['metric'] == 'accuracy':
        metrics = ['accuracy']
    elif params['metric'] == 'sparse':
        metrics = [tensorflow.metrics.SparseCategoricalAccuracy()]
    #elif params['metric'] == 'tn':
        #metrics = [tensorflow.metrics.TrueNegatives(),tensorflow.metrics.TruePositives()]
    model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=metrics)  # , AUC(multi_label=False)])

    return model

def cmvals(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn = cm[0, 0]
    fp = cm[0, 1]
    fn = cm[1, 0]
    tp = cm[1, 1]
    return tn, fp, fn, tp

def nn_fit_and_predict(params, X_pd_tr = None, y_pd_tr = None, X_pd_tst = None, y_pd_tst = None, testdate = None, metrics = None):

    modelfolder = 'models'

    cnt = 0
    print("NN params : %s" % params)


    modelfname = "".join([c for c in json.dumps(params) if re.match(r'\w', c)])

    aucmetric = tensorflow.metrics.AUC()

    if X_pd_tr is not None and y_pd_tr is not None and not os.path.exists(os.path.join(modelfolder, modelfname)):
        if len(params['feature_drop'])>0:
            X_pd_tr = X_pd_tr.drop(columns=[c for c in X_pd_tr.columns if any([fd in c for fd in params['feature_drop']])])

        X_train = X_pd_tr.values
        y_train = y_pd_tr.values
        y_scores = None

        print("Fitting ...")
        # print("TRAIN:", train_index, "TEST:", test_index)
        y_train = y_train[:,0]
        model = create_NN_model(params, X_train)
        es = EarlyStopping(monitor='loss', patience=10, min_delta=0.002)
        start_time = time.time()
        res = model.fit(X_train, y_train, batch_size=512, epochs=max_epochs, verbose=0,\
                        callbacks=[es], class_weight=params['class_weights'])

        print("Fit time (min): %s"%((time.time() - start_time)/60.0))
        start_time = time.time()
        es_epochs = len(res.history['loss'])
        model.save(os.path.join(modelfolder,modelfname))

        '''training set metrics'''
        '''
        loss_train, acc_train = model.evaluate(X_train, y_train, batch_size=512, verbose=0)
        y_pred = model.predict_classes(X_train)
        y_scores = model.predict(X_train)
        aucmetric.update_state(y_train, y_scores[:,1])
        auc_train = float(aucmetric.result())

        acc_1_train = accuracy_score(y_train, y_pred)
        acc_0_train = accuracy_score(1-y_train, 1-y_pred)

        prec_1_train = precision_score(y_train, y_pred)
        prec_0_train = precision_score(1-y_train, 1-y_pred)

        rec_1_train = recall_score(y_train, y_pred)
        rec_0_train = recall_score(1-y_train, 1-y_pred)

        f1_1_train = f1_score(y_train, y_pred)
        f1_0_train = f1_score(1-y_train, 1-y_pred)
        '''

    '''test set metrics'''

    if not os.path.exists(os.path.join(modelfolder, modelfname)):
        print('Model %s not found %s'%modelfname)
        return
    if X_pd_tst is None or y_pd_tst is None:
        print('No test data')
        return

    start_time = time.time()

    model = models.load_model(os.path.join(modelfolder, modelfname))
    model = models.load_model(os.path.join(modelfolder, modelfname))

    #if params['feature_drop']:
    #    X_pd_tst = X_pd_tst.drop(columns=[c for c in X_pd_tst.columns if params['feature_drop'] in c])
    if len(params['feature_drop']) > 0:
        X_pd_tst = X_pd_tst.drop(columns=[c for c in X_pd_tst.columns if any([fd in c for fd in params['feature_drop']])])

    X_test = X_pd_tst.values
    y_test = y_pd_tst.values
    y_test = y_test[:, 0]

    start_predict = time.time()
    #loss_test, acc_test = model.evaluate(X_test, y_test, batch_size=512, verbose=0)
    #y_pred_1 = model.predict_classes(X_test)
    y_scores = model.predict(X_test)
    predict_class = lambda p : int(round(p))
    predict_class_v = np.vectorize(predict_class)
    y_pred = predict_class_v(y_scores[:,1])

    aucmetric.update_state(y_test, y_scores[:,1])
    auc_val = float(aucmetric.result())

    acc_1_test = accuracy_score(y_test, y_pred)
    acc_0_test = accuracy_score(1 - y_test, 1 - y_pred)

    prec_1_test = precision_score(y_test, y_pred)
    prec_0_test = precision_score(1 - y_test, 1 - y_pred)

    rec_1_test = recall_score(y_test, y_pred)
    rec_0_test = recall_score(1 - y_test, 1 - y_pred)

    f1_1_test = f1_score(y_test, y_pred)
    f1_0_test = f1_score(1 - y_test, 1 - y_pred)

    tn1, fp1, fn1, tp1 = cmvals(y_test, y_pred)
    tn0, fp0, fn0, tp0 = cmvals(1-y_test, 1-y_pred)

    print("Test metrics time (min): %s"%((time.time() - start_predict)/60.0))
    print("Recall 1 : %s, Recall 0 : %s" % (rec_1_test, rec_0_test))

    testdate = 'metrics' if not testdate else testdate
    if metrics is not None:
        metrics.append({'date' : testdate,
             'accuracy test': acc_1_test, 'precision 1 test': prec_1_test, 'recall 1 test' : rec_1_test,
             'f1-score 1 test': f1_1_test, 'True Negative 1': tn1, 'False Positive 1': fp1, 'False Negative 1': fn1,'True Positive 1': tp1,
             'accuracy 0 test': acc_0_test,'precision 0 test': prec_0_test, 'recall 0 test': rec_0_test,
              'f1-score 0 test': f1_0_test,'auc test': auc_val,
             'True Negative 0': tn0, 'False Positive 0': fp0, 'False Negative 0': fn0, 'True Positive 0': tp0,
             'predict time':  (time.time() - start_time)/60.0 })

    return y_scores

def calcminmaxstats(dstestfiles, statfname):
    if not os.path.exists(os.path.join('stats', 'featurestats.json')):
        stats = {}
    else:
        with open(statfname, 'r') as statfile:
            stats = json.loads(statfile.read())
    for X_pd, y_pd, tdate in load_datasets(dstrainfile, statfname=statfname):
        i=1
    for dstestfile in dstestfiles:
        for X_pd, y_pd, tdate in load_datasets(dstestfile, calcstats = stats):
            i=1
    with open(os.path.join('stats','featurestats.json'),'w') as statfile:
        statfile.write(json.dumps(stats, default=npconv))
    return stats

def recall(tp,fn):
    if tp+fn == 0:
        return 0
    return tp/(tp+fn)

def precision(tp,fp):
    if tp+fp == 0:
        return 0
    return tp/(tp+fp)

def accuracy(tp,tn, fp, fn):
    if tp+fp+fp+fn == 0:
        return 0
    return (tp+tn)/(tp+fp+fp+fn)

def f1(tp,fp,fn):
    if recall(tp,fn)+precision(tp,fp) == 0:
        return 0
    return 2*recall(tp,fn)*precision(tp,fp)/(recall(tp,fn)+precision(tp,fp))


dstestfiles, dstrainfile, space, max_epochs, checkunnorm, savescores = space_test.create_space()

statfname = os.path.join('stats', 'featurestats.json')

if not os.path.exists(os.path.join('stats', 'featurestats.json')):
    stats = calcminmaxstats(dstestfiles, statfname)
else:
    with open(os.path.join('stats', 'featurestats.json'), 'r') as statfile:
        stats = json.loads(statfile.read())

for X_pd, y_pd, tdate in load_datasets(dstrainfile, statfname = statfname):
    nn_fit_and_predict(space, X_pd_tr = X_pd, y_pd_tr = y_pd, X_pd_tst = None, y_pd_tst = None)


allfilemetrics = None
modelfname = "".join([ch for ch in json.dumps(space) if re.match(r'\w', ch)])
for dstestfile in dstestfiles:
    metrics = []
    dsetfolder, dsreadysuffix, dsunnormsuffix, dsfile, dsetfolder, dsready = filenames(dstestfile)
    #flist = [fn for fn in os.listdir(dsetfolder) if os.path.basename(dsready) in fn and dsunnormsuffix not in fn]
    for X_pd, y_pd, tdate in load_datasets(dstestfile, perdate=True, statfname = statfname, checkunnorm = checkunnorm):
        y_scores = nn_fit_and_predict(space, X_pd_tr = None, y_pd_tr = None, X_pd_tst = X_pd, y_pd_tst = y_pd, testdate = tdate, metrics = metrics)
        month = str(tdate)[:6]
        flist = [fn for fn in os.listdir(dsetfolder) if os.path.basename(dsready) in fn and dsunnormsuffix not in fn]
        if y_scores is not None and savescores:
            fn = os.path.join(dsetfolder,[f for f in flist if str(tdate) in f][0])
            modelparams = ''.join([str(space[p]) for p in space])
            model_scorename = 'scores '+''.join([ch for ch in modelparams if re.match(r'\w', ch)])
            scores = pd.Series(y_scores[:,1], name = model_scorename)
            featdf = pd.read_csv(fn)
            #featdf = featdf.astype({'id': 'int32'})
            score_cols = [c for c in featdf.columns if 'scores' in c]
            if score_cols is not None and len(score_cols)>0:
                for c in score_cols:
                    if c == model_scorename:
                        featdf = featdf.drop([c], axis=1)
            featdf = pd.concat([featdf, scores], axis=1)
            featdf.to_csv(fn, index=False)

    pdmetrics = pd.DataFrame(metrics)
    if allfilemetrics is None:
        allfilemetrics = pd.DataFrame(columns=pdmetrics.columns)
    pdmetrics = pdmetrics.append(pd.Series(name = 'sums'))
    sumscols = [ 'True Negative 1', 'False Positive 1', 'False Negative 1', 'True Positive 1',\
                 'True Negative 0', 'False Positive 0', 'False Negative 0', 'True Positive 0', 'predict time']
    for c in sumscols:
        pdmetrics[c]['sums'] = pdmetrics[c].sum()

    pdmetrics['accuracy test']['sums'] = accuracy(pdmetrics['True Positive 1']['sums'],pdmetrics['True Negative 1']['sums'],\
                                                  pdmetrics['False Positive 1']['sums'], pdmetrics['False Negative 1']['sums'])
    pdmetrics['precision 1 test']['sums'] = precision(pdmetrics['True Positive 1']['sums'], pdmetrics['False Positive 1']['sums'])
    pdmetrics['precision 0 test']['sums'] = precision(pdmetrics['True Positive 0']['sums'], pdmetrics['False Positive 0']['sums'])
    pdmetrics['recall 1 test']['sums'] = recall(pdmetrics['True Positive 1']['sums'],pdmetrics['False Negative 1']['sums'])
    pdmetrics['recall 0 test']['sums'] = recall(pdmetrics['True Positive 0']['sums'],pdmetrics['False Negative 0']['sums'])
    pdmetrics['f1-score 1 test']['sums'] = f1(pdmetrics['True Positive 1']['sums'], pdmetrics['False Positive 1']['sums'], pdmetrics['False Negative 1']['sums'])
    pdmetrics['f1-score 0 test']['sums'] = f1(pdmetrics['True Positive 0']['sums'], pdmetrics['False Positive 0']['sums'], pdmetrics['False Negative 0']['sums'])
    pdmetrics['date']['sums'] = month
    res_pref = 'results/test_results_'
    #res_base = res_pref+os.path.basename(dstestfile)[:-4]+modelfname
    res_base = "%s%s_%s"%(res_pref, month, modelfname)
    cnt = 1
    while os.path.exists('%s_%d.csv' % (res_base, cnt)):
        cnt += 1
    pdmetrics.to_csv('%s_%d.csv' % (res_base, cnt), index=False)
    allfilemetrics = allfilemetrics.append(pdmetrics.loc['sums'].to_dict(), ignore_index=True)

#modelfname = "".join([ch for ch in json.dumps(space) if re.match(r'\w', ch)])
allfilemetrics.to_csv('%s_%d.csv' % (res_pref+modelfname, cnt), index=False)
