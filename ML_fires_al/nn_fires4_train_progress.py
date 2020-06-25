#!/usr/bin/env python

from pandas import read_csv
from sklearn.model_selection import train_test_split
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
import hyperopt

# load the dataset
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

# split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=0.10)
# determine the number of input features
n_features = X_train.shape[1]

# define model
model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(n_features,)))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
model.add(Dense(2, activation='softmax'))

# compile the model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
from tensorflow.keras.optimizers import Adam

adam = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

es = EarlyStopping(monitor='loss', mode='min', patience=20)
# %load_ext tensorboard
# %reload_ext tensorboard
# %tensorboard --logdir C:\Users\User\Documents\codeprojects\FFP\logs\1
# file_writer = tensorflow.summary.FileWriter('C:\\Users\\User\\Documents\\codeprojects\\FFP\\logs\\', sess.graph)
# log_dir = os.path.join('.\\logs\\s2')
# tb = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True, profile_batch = 100000000)
# model.fit(X_train[:,1:], y_train, epochs=100, batch_size=1000, callbacks = [es, tb])
# model.fit(X_train, y_train, epochs=100, batch_size=1000, callbacks = [es, tb])

# loss, acc = model.evaluate(X_test, y_test, verbose=0)
# print('Test Accuracy: %.3f' % acc)

# preds = np.argmax(model.predict(X_test), axis = 1)


er_cv_pl = []
er_train_pl = []
acc_cv_pl = []
acc_train_pl = []
xvals = []
for isize in range(5, 101, 5):

    xvals.append(isize)
    if isize == 100:
        X_part, y_part = X_, y_
    else:
        X_part, _X_test, y_part, _y_test = train_test_split(X_, y_,
                                                            test_size=(100 - isize) / 100)  # 20% hold out for testing

    X_train, X_cv, y_train, y_cv = train_test_split(X_part, y_part, test_size=0.10)

    # train classifier
    nn = model.fit(X_train, y_train, epochs=100, batch_size=512, callbacks=[es], verbose=0)
    # er_cv, acc_cv = model.evaluate(X_test, y_test, verbose=0)
    # er_train, acc_train = model.evaluate(X_train, y_train, verbose=0)

    # run on test set
    '''
    y_pred_train = model.predict(X_train)
    y_pred_train_bin = (y_pred_train[:,1]+0.5).astype(int)
    er_train = 1 - accuracy_score(y_train, y_pred_train_bin)
    er_train_pl.append(er_train)
    y_pred_cv = model.predict(X_cv)
    y_pred_cv_bin = (y_pred_cv[:,1]+0.5).astype(int)
    er_cv = 1 - accuracy_score(y_cv, y_pred_cv_bin)
    er_cv_pl.append(er_cv)
    '''

    losstr, acctr = model.evaluate(X_train, y_train, verbose=0)
    losscv, acccv = model.evaluate(X_cv, y_cv, verbose=0)
    er_train_pl.append(losstr)
    acc_train_pl.append(acctr)
    er_cv_pl.append(losscv)
    acc_cv_pl.append(acccv)

    print('Data set size: %2.0f%%. Tr. Loss: %5.2f, C.V. Loss: %5.2f Tr. Acc: %5.2f, C.V. Acc.: %5.2f'\
          % (isize, losstr, losscv, acctr, acccv))

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