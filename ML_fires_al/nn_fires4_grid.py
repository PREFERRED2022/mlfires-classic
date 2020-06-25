from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
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
import os
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from matplotlib import axis
import itertools as it


def normalized_values(y, dfmax, dfmin, dfmean, dfstd, t=None):
    if not t:
        a = (y - dfmin) / (dfmax - dfmin)
        return (a)
    elif t == 'std':
        a = (y - dfmean) / dfstd
        return (a)
    elif t == 'no':
        return y


def normalize_dataset(X_unnorm_int, norm_type=None):
    X = DataFrame()
    for c in X_unnorm_int.columns:
        print(c)
        dfmax = X_unnorm_int[c].max()
        dfmin = X_unnorm_int[c].min()
        dfmean = X_unnorm_int[c].mean()
        dfstd = X_unnorm_int[c].std()
        X[c] = X_unnorm_int.apply(lambda x: normalized_values(x[c], dfmax, dfmin, dfmean, dfstd, norm_type), axis=1)
    return X


def convtoindex(y, lu_dict):
    return (lu_dict[y])


def indexdict(dfcol):
    lu = list(dfcol.unique())
    lu_dict = {x: lu.index(x) + 1 for x in lu}
    return lu_dict


def index_string_values(X_unnorm, str_classes):
    indexdicts = {}
    for str_class in str_classes:
        indexdicts[str_class] = indexdict(X_unnorm[str_class])
    X_unnorm_int = X_unnorm.copy()
    for c in str_classes:
        print(c)
        X_unnorm_int[c] = X_unnorm.apply(lambda x: convtoindex(x[c], indexdicts[c]), axis=1)
    return X_unnorm_int


def gridsearchAS(paramgrid):
    # allparams = sorted(paramgrid)
    combinations = it.product(*(paramgrid[p] for p in paramgrid))
    # print(list(combinations))
    scores = []
    for c in combinations:
        print(c)
        model.fit(X_train, y_train, epochs=c[1], batch_size=c[0], verbose=0)
        losstr, acctr = model.evaluate(X_train, y_train, verbose=0)
        losstst, acctst = model.evaluate(X_test, y_test, verbose=0)
        scores.append(list(c) + [losstr, acctr, losstst, acctst])
    return scores


def plotstats(scores):
    plt.figure()
    for var in range(2, 4):
        labelvar = 'Loss' if var == 2 else 'Accuracy'
        for i in range(0, 2):
            label = 'Train ' + labelvar if i == 0 else 'Test ' + labelvar
            plt.plot([s[1] for s in scores], [s[var + i * 2] for s in scores],
                     label=label)
        plt.legend(loc="upper center")
    plt.title("NN loss, accuracy vs epochs")
    plt.ylabel('Loss/Accuracy')
    plt.xlabel('Epochs')
    ax=plt.gca()
    ax.yaxis.grid()
    #plt.savefig('/home/sgirtsou/Documents/ML-dataset_newLU/csvs_withfire_results/epochs_%s_%s.png' % (
    #scores[-1][1], scores[-1][0]))
    #plt.show()
    #plt.close()


df = read_csv('/home/sgirtsou/Documents/ML-dataset_newLU/training_dataset.csv')
df = df.dropna()

df_part = df[
    ['id', 'max_temp', 'min_temp', 'mean_temp', 'res_max', 'dir_max', 'dom_vel', 'dom_dir', 'rain_7days', 'Corine',
     'Slope', 'DEM', 'Curvature', 'Aspect', 'ndvi', 'fire']].copy()

X_unnorm, y_int = df_part[
                      ['max_temp', 'min_temp', 'mean_temp', 'res_max', 'dir_max', 'dom_vel', 'dom_dir', 'rain_7days',
                       'Corine', 'Slope', 'DEM', 'Curvature', 'Aspect', 'ndvi']], df_part['fire']

str_classes = ['Corine']
X_unnorm_int = index_string_values(X_unnorm, str_classes)
X = normalize_dataset(X_unnorm_int, 'std')

# y = to_categorical(y_int)
y = y_int

X_ = X.values
y_ = y.values
y.shape, X.shape, type(X_), type(y_)

# split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=0.10)
# determine the number of input features
n_features = X_train.shape[1]

# define model
model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(n_features,)))
model.add(Dense(8, activation='relu'))
model.add(Dense(0, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
model.add(Dense(2, activation='softmax'))

# compile the model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
from tensorflow.keras.optimizers import Adam

adam = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print(model.metrics_names)

# fit the model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow import summary
import os

es = EarlyStopping(monitor='loss', mode='min', patience=20)
# %load_ext tensorboard
# %reload_ext tensorboard
# %tensorboard --logdir C:\Users\User\Documents\codeprojects\FFP\logs\1
# file_writer = tensorflow.summary.FileWriter('C:\\Users\\User\\Documents\\codeprojects\\FFP\\logs\\', sess.graph)
log_dir = '/home/sgirtsou/Documents/ML-dataset_newLU/csvs_withfire_results/NN_results/logs'
tb = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True, profile_batch=100000000)
# model.fit(X_train[:,1:], y_train, epochs=100, batch_size=1000, callbacks = [es, tb])
# model.fit(X_train, y_train, epochs=100, batch_size=1000)#, callbacks=[es, tb])
batch_size = [512]
epochs = range(10, 311, 20)
param_grid = dict(batch_size=batch_size, epochs=epochs)
scores = gridsearchAS(param_grid)
# evaluate the model
# loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(scores)
plotstats(scores)

'''
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import roc_auc_score
import sys

X = []
y = []
X_val = []
y_val = []

space = {'choice': hp.choice('num_layers',
                    [ {'layers':'two', },
                    {'layers':'three',
                    'units3': hp.uniform('units3', 64,1024),
                    'dropout3': hp.uniform('dropout3', .25,.75)}
                    ]),

            'units1': hp.uniform('units1', 64,1024),
            'units2': hp.uniform('units2', 64,1024),

            'dropout1': hp.uniform('dropout1', .25,.75),
            'dropout2': hp.uniform('dropout2',  .25,.75),

            'batch_size' : hp.uniform('batch_size', 28,128),

            'nb_epochs' :  100,
            'optimizer': hp.choice('optimizer',['adadelta','adam','rmsprop']),
            'activation': 'relu'
        }

def f_nn(params):
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation
    from keras.optimizers import Adadelta, Adam, rmsprop

    print ('Params testing: ', params)
    model = Sequential()
    model.add(Dense(output_dim=params['units1'], input_dim = X.shape[1]))
    model.add(Activation(params['activation']))
    model.add(Dropout(params['dropout1']))

    model.add(Dense(output_dim=params['units2'], init = "glorot_uniform"))
    model.add(Activation(params['activation']))
    model.add(Dropout(params['dropout2']))

    if params['choice']['layers']== 'three':
        model.add(Dense(output_dim=params['choice']['units3'], init = "glorot_uniform"))
        model.add(Activation(params['activation']))
        model.add(Dropout(params['choice']['dropout3']))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=params['optimizer'])

    model.fit(X, y, nb_epoch=params['nb_epochs'], batch_size=params['batch_size'], verbose = 0)

    pred_auc =model.predict_proba(X_val, batch_size = 128, verbose = 0)
    acc = roc_auc_score(y_val, pred_auc)
    print('AUC:', acc)
    sys.stdout.flush()
    return {'loss': -acc, 'status': STATUS_OK}


trials = Trials()
best = fmin(f_nn, space, algo=tpe.suggest, max_evals=50, trials=trials)
print('best: ', best)
'''