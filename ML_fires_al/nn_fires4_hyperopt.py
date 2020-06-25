#!/usr/bin/env python
from hyperopt import Trials, fmin, tpe, hp
from pandas import read_csv
from sklearn.model_selection import train_test_split, KFold
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
from sklearn.metrics import accuracy_score


num_folds=5
kf = KFold(n_splits=num_folds)
random_state=42

# load the dataset
def load_dataset():
    df = read_csv('/home/sgirtsou/Documents/ML-dataset_newLU/training_dataset.csv')
    df = df.dropna()
    df.columns
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

    df_part = df[
        ['id', 'max_temp', 'min_temp', 'mean_temp', 'res_max', 'dir_max', 'dom_vel', 'dom_dir', 'rain_7days', 'Corine',
         'Slope', 'DEM', 'Curvature', 'Aspect', 'ndvi', 'fire']].copy()

    X_unnorm, y_int = df_part[
                          ['max_temp', 'min_temp', 'mean_temp', 'res_max', 'dir_max', 'dom_vel', 'dom_dir', 'rain_7days',
                           'Corine', 'Slope', 'DEM', 'Curvature', 'Aspect', 'ndvi']], df_part['fire']

    str_classes = ['Corine']
    X_unnorm_int = normdataset.index_string_values(X_unnorm, str_classes)
    X = normdataset.normalize_dataset(X_unnorm_int, 'std')

    y = y_int

    X_ = X.values
    y_ = y.values
    return X_, y_

X,y = load_dataset()

def create_NN_model(params, X=X):
    # define model
    model = Sequential()
    n_features = X.shape[1]
    model.add(Dense(params['layer_1_nodes'], activation='relu', input_shape=(n_features,)))
    for i in range(2, int(params['n_internal_layers'])+2):
        model.add(Dense(int(params['layer_'+str(i)+'_nodes']), activation='relu'))
        # model.add(Dense(1, activation='sigmoid'))
    model.add(Dense(2, activation='softmax'))

    # compile the model
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    from tensorflow.keras.optimizers import Adam

    adam = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def create_NN_model_simple(params):
    # define model
    model = Sequential()
    model.add(Dense(params['layer_1_nodes'], activation='relu', input_shape=(n_features,)))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    # compile the model
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    from tensorflow.keras.optimizers import Adam

    adam = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def nnfit(params, cv=kf, X=X, y=y):
    # the function gets a set of variable parameters in "param"
    params = {'n_internal_layers': params['n_internal_layers'],
              'layer_1_nodes': params['layer_1_nodes'],
              'layer_2_nodes': params['layer_2_nodes'],
              'layer_3_nodes': params['layer_3_nodes'],
              'layer_4_nodes': params['layer_4_nodes']}

    # we use this params to create a new LGBM Regressor
    # model = LGBMRegressor(random_state=random_state, **params)
    model = create_NN_model(params)

    # and then conduct the cross validation with the same folds as before
    #score = -cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1).mean()

    losses=[]
    accuracies=[]
    cnt=0
    print("NN params : %s"%params)
    for train_index, test_index in cv.split(X):
        cnt+=1
        print("Fitting Fold %d"%cnt)
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train, epochs=1, batch_size=512, verbose=0)
        loss, acc = model.evaluate(X_test, y_test, batch_size=512, verbose=0)
        losses.append(loss)
        accuracies.append(acc)
    meanloss = sum(losses)/len(losses)
    meanacc = -sum(accuracies)/len(accuracies)
    print(meanacc)
    return meanacc


def nnfit_simple(params):
    # the function gets a set of variable parameters in "param"
    params = {'layer_1_nodes': params['layer_1_nodes']}

    # we use this params to create a new LGBM Regressor
    # model = LGBMRegressor(random_state=random_state, **params)
    model = create_NN_model_simple(params)

    # and then conduct the cross validation with the same folds as before
    #score = -cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1).mean()


    #nn = model.fit(X_train, y_train, epochs=250, batch_size=512, verbose=0)
    loss=1
    return loss



# split into train and test datasets
#X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=0.10)
# determine the number of input features

space={'n_internal_layers': hp.quniform('n_internal_layers', 0, 3, 1),
       'layer_1_nodes': hp.quniform('layer_1_nodes', 2, 40, 10),
       'layer_2_nodes': hp.quniform('layer_2_nodes', 2, 40, 10),
       'layer_3_nodes': hp.quniform('layer_3_nodes', 2, 40, 10),
       'layer_4_nodes': hp.quniform('layer_4_nodes', 2, 40, 10)
      }


trials = Trials()

best=fmin(fn=nnfit, # function to optimize
          space=space,
          algo=tpe.suggest, # optimization algorithm, hyperotp will select its parameters automatically
          max_evals=4, # maximum number of iterations
          trials=trials, # logging
          rstate=np.random.RandomState(random_state) # fixing random state for the reproducibility
         )

print("Best {:.3f} params {}".format(-nnfit(best), best))
'''
plt.plot(xvals, er_train_pl, label="Loss Train")
plt.plot(xvals, er_cv_pl, label="Loss Cross Val.")
plt.plot(xvals, acc_train_pl, label="Acc. Train")
plt.plot(xvals, acc_cv_pl, label="Acc. Cross Val.")
plt.legend(loc="upper center")
# plt.axis([0, 100, 0, 1])
plt.ylabel('Training vs Cross Valid. ')
plt.xlabel('Dataset size')
ax = plt.gca()
ax.yaxis.grid()
plt.savefig('/home/sgirtsou/Documents/ML-dataset_newLU/csvs_withfire_results/nn_progress_2.png')
# plt.show()
# plt.close()
'''