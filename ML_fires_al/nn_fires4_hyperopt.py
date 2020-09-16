#!/usr/bin/env python
from hyperopt import Trials, fmin, tpe, hp, STATUS_OK
from pandas import read_csv
from sklearn.model_selection import train_test_split, KFold, GroupKFold
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
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


num_folds=10
#kf = KFold(n_splits=num_folds, shuffle=True)
kf = GroupKFold(n_splits=num_folds)
random_state=42

# load the dataset
def load_dataset():
    #df = read_csv('/home/sgirtsou/Documents/ML-dataset_newLU/training_dataset.csv')
    df = pd.read_csv('/home/sgirtsou/Documents/ML-dataset_newLU/dataset_ndvi_lu.csv')
    df = df.dropna()
    # df.loc[df.fire == 1, 'res_max'].std()
    # df.loc[(df.fire == 0) & (df.max_temp > 0.5), 'res_max'].std()
    run_all_day = False
    if run_all_day:
        df_greece_d = read_csv("C:/Users/User/Documents/projects/FFP/one day 2018 dataset/dataset4_greece_colnam.csv")
        df_greece = df_greece_d[df_greece_d.max_temp != '--']
        print("df_greece cols", df_greece.columns)

        # exclude from dataset fires of day
        firesofdaycells = df_greece[df_greece.fire == 1]['id'].tolist()
        # daycellsindf = df1[df1['id'].isin(firesofdaycells) & df1['fire']==1]['id'].tolist()
        # df = df[~df['id'].isin(firesofdaycells) & df['fire']==1]
        df = df[~df['id'].isin(firesofdaycells)]
        df.shape


    print(df.columns)

    df.columns = ['id', 'firedate_x', 'max_temp', 'min_temp', 'mean_temp', 'res_max',
                  'dir_max', 'dom_vel', 'dom_dir', 'rain_7days', 'Corine', 'Forest',
                  'fire', 'firedate_g', 'firedate_y', 'tile', 'max_temp_y', 'DEM',
                  'Slope', 'Curvature', 'Aspect', 'image', 'ndvi']
    df_part = df[
        ['id', 'max_temp', 'min_temp', 'mean_temp', 'res_max', 'dir_max', 'dom_vel', 'dom_dir', 'rain_7days', 'Corine',
         'Slope', 'DEM', 'Curvature', 'Aspect', 'ndvi', 'fire']]

    X_unnorm, y_int = df_part[
                          ['max_temp', 'min_temp', 'mean_temp', 'res_max', 'dir_max', 'dom_vel', 'dom_dir', 'rain_7days',
                           'Corine', 'Slope', 'DEM', 'Curvature', 'Aspect', 'ndvi']], df_part['fire']

    # categories to binary
    Xbindomdir = pd.get_dummies(X_unnorm['dom_dir'].round())
    del Xbindomdir[0]
    ddircols=[]
    for i in range(1,9):
        ddircols.append('binDDIR_%d'%i)
    Xbindomdir.columns = ddircols
    Xbindirmax = pd.get_dummies(X_unnorm['dir_max'].round())
    del Xbindirmax[0]
    dmaxcols=[]
    for i in range(1, 9):
        dmaxcols.append('binMDIR_%d' % i)
    Xbindirmax.columns = dmaxcols


    Xbincorine = pd.get_dummies(X_unnorm['Corine'])
    corcols=['bin'+str(c) for c in Xbincorine.columns]
    Xbincorine.columns = corcols
    X_unnorm = pd.concat([X_unnorm, Xbindomdir, Xbindirmax, Xbincorine], axis=1)
    del X_unnorm['Corine']
    del X_unnorm['dom_dir']
    del X_unnorm['dir_max']

    # X = normalize_dataset(X_unnorm, 'std')

    str_classes = ['Corine']
    #X_unnorm_int = normdataset.index_string_values(X_unnorm, str_classes)
    #X = normdataset.normalize_dataset(X_unnorm_int, 'std')

    X = normdataset.normalize_dataset(X_unnorm)
    y = y_int

    X_ = X.values
    y_ = y.values

    groupspd = df['firedate_g']
    groups = groupspd.values

    return X_, y_, groups

X,y,groups = load_dataset()


def create_NN_model(params, X):
    # define model
    model = Sequential()
    n_features = X.shape[1]
    intlayers = int(params['n_internal_layers'][0])
    model.add(Dense(params['n_internal_layers'][1]['layer_1_'+str(intlayers)+'_nodes'], activation='relu', input_shape=(n_features,)))
    for i in range(2, intlayers+2):
        model.add(Dense(int(params['n_internal_layers'][1]['layer_'+str(i)+'_'+str(intlayers)+'_nodes']), activation='relu'))
        # model.add(Dense(1, activation='sigmoid'))
    model.add(Dense(2, activation='softmax'))

    # compile the model
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    from tensorflow.keras.optimizers import Adam

    adam = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def nnfit(params, cv=kf, X=X, y=y, groups=groups):
    # the function gets a set of variable parameters in "param"
    '''
    params = {'n_internal_layers': params['n_internal_layers'][0],
              'layer_1_nodes': params['n_internal_layers'][1],
              'layer_2_nodes': params['layer_2_nodes'],
              'layer_3_nodes': params['layer_3_nodes'],
              'layer_4_nodes': params['layer_4_nodes']}
              '''

    metrics=[]
    cnt=0
    print("NN params : %s"%params)
    for train_index, test_index in cv.split(X,y, groups):
        cnt+=1
        print("Fitting Fold %d"%cnt)
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = create_NN_model(params, X)
        model.fit(X_train, y_train, epochs=250, batch_size=512, verbose=0)
        loss_test, acc_test = model.evaluate(X_test, y_test, batch_size=512, verbose=0)
        y_pred = model.predict_classes(X_test)
        #yall = np.vstack((y_test, y_pred))
        prec_test = precision_score(y_test, y_pred)
        rec_test = recall_score(y_test, y_pred)
        f1_test = f1_score(y_test, y_pred)
        loss_train, acc_train = model.evaluate(X_train, y_train, batch_size=512, verbose=0)
        y_pred = model.predict_classes(X_train)
        prec_train = precision_score(y_train, y_pred)
        rec_train = recall_score(y_train, y_pred)
        f1_train = f1_score(y_train, y_pred)
        metrics.append({'loss test':loss_test, 'loss train':loss_train, 'accuracy test':acc_test, 'accuracy train':acc_train,
                        'precision test':prec_test,  'precision train':prec_train, 'recall test':rec_test, 'recall train':rec_train,
                        'f1-score test':f1_test, 'f1-score train':f1_train})


    mean_metrics={}
    for m in metrics[0]:
        mean_metrics[m] = sum(item.get(m, 0) for item in metrics) / len(metrics)
    print('Mean recall (on test) : %s'%mean_metrics['recall test'])

    return {
        'loss': -mean_metrics['recall test'],
        'status': STATUS_OK,
        # -- store other results like this
        #'eval_time': time.time(),
        'metrics': mean_metrics,
        # -- attachments are handled differently
        #'attachments':
        #    {'time_module': pickle.dumps(time.time)}
    }

'''
space={'n_internal_layers': hp.quniform('n_internal_layers', 0, 3, 1),
       'layer_1_nodes': hp.quniform('layer_1_nodes', 5, 40, 5),
       'layer_2_nodes': hp.quniform('layer_2_nodes', 5, 40, 5),
       'layer_3_nodes': hp.quniform('layer_3_nodes', 5, 40, 5),
       'layer_4_nodes': hp.quniform('layer_4_nodes', 5, 40, 5)
      }
'''

space= {'n_internal_layers': hp.choice('n_internal_layers',
           [
               (0, {'layer_1_0_nodes':hp.quniform('layer_1_0_nodes', 10, 50, 10)}),
               #(1, {'layer_1_1_nodes':hp.quniform('layer_1_1_nodes', 10, 50, 10), 'layer_2_1_nodes':hp.quniform('layer_2_1_nodes', 10, 50, 10)}),
               #(2, {'layer_1_2_nodes':hp.quniform('layer_1_2_nodes', 10, 50, 10), 'layer_2_2_nodes':hp.quniform('layer_2_2_nodes', 10, 50, 10), 'layer_3_2_nodes':hp.quniform('layer_3_2_nodes', 10, 50, 10)})
           ]
       )}

'''
space={'n_internal_layers': hp.choice('n_internal_layers', [2]),
       'layer_1_nodes': hp.choice('layer_1_nodes', [20]),
       'layer_2_nodes': hp.choice('layer_2_nodes', [10]),
       'layer_3_nodes': hp.choice('layer_3_nodes', [5]),
       'layer_4_nodes': hp.choice('layer_4_nodes', [20])
      }
'''


trials = Trials()

best=fmin(fn=nnfit, # function to optimize
          space=space,
          algo=tpe.suggest, # optimization algorithm, hyperotp will select its parameters automatically
          max_evals=10, # maximum number of iterations
          trials=trials, # logging
          rstate=np.random.RandomState(random_state) # fixing random state for the reproducibility
         )

pd_opt = pd.DataFrame(columns=list(trials.trials[0]['result']['metrics'].keys()))
for t in trials:
    pdrow = t['result']['metrics']
    pdrow['params'] = str(t['misc']['vals'])
    pd_opt = pd_opt.append(pdrow, ignore_index=True)

hyp_res_base = 'hyperopt_results_'
cnt=1
while os.path.exists('%s%d.csv'%(hyp_res_base, cnt)):
    cnt+=1
pd_opt.to_csv('%s%d.csv'%(hyp_res_base, cnt))
