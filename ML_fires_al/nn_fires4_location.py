from pandas import read_csv
from sklearn.model_selection import train_test_split
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
import normdataset

df = read_csv('/home/sgirtsou/Documents/ML-dataset_newLU/training_dataset.csv')
df = df.dropna()

df_part = df[['x','y', 'fire']]

# split into input and output columns

X_unnorm, y_int = df_part[['x','y']], df_part['fire']

# ensure all data are floating point values

print(X_unnorm)

X = normdataset.normalize_dataset(X_unnorm, 'std')
print(X)

print(y_int)

#y = to_categorical(y_int)
y=y_int

X_ = X.values
y_ = y.values
y.shape, X.shape, type(X_), type(y_)

# split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=0.10)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# determine the number of input features
n_features = X_train.shape[1]
#n_features = X_train.shape[1]
n_features
# define model

np.count_nonzero(y_test == 0),len(y_test)

type(y_train)

# define model
model = Sequential()
model.add(Dense(400, activation='relu', input_shape=(n_features,)))
#model.add(Dense(1, activation='sigmoid'))
model.add(Dense(2, activation='softmax'))

# compile the model
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
from tensorflow.keras.optimizers import Adam
adam = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# fit the model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow import summary
import os
es = EarlyStopping(monitor='loss', mode='min', patience = 20)
# %load_ext tensorboard
# %reload_ext tensorboard
# %tensorboard --logdir C:\Users\User\Documents\codeprojects\FFP\logs\1
# file_writer = tensorflow.summary.FileWriter('C:\\Users\\User\\Documents\\codeprojects\\FFP\\logs\\', sess.graph)
log_dir = os.path.join('./logs')
tb = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True, profile_batch = 100000000)
#model.fit(X_train[:,1:], y_train, epochs=100, batch_size=1000, callbacks = [es, tb])
model.fit(X_train, y_train, epochs=250, batch_size=512, callbacks = [es, tb])

# evaluate the model
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print('Test Accuracy: %.3f' % acc)

#make a prediction

#row = [1,0,0.99539,-0.05889,0.85243,0.02306,0.83398,-0.37708,1,0.03760,0.85243,-0.17755,0.59755,-0.44945,0.60536,-0.38223,0.84356,-0.38542,0.58212,-0.32192,0.56971,-0.29674,0.36946,-0.47357,0.56811,-0.51171,0.41078,-0.46168,0.21266,-0.34090,0.42267,-0.54487,0.18641,-0.45300]
#yhat = model.predict([row])
#print('Predicted: %.3f' % yhat)
#display(yhat)

preds = np.argmax(model.predict(X_test), axis = 1)

fp = np.where(preds - y_test == 1)
fn = np.where(preds - y_test == -1)

mypath = '/home/sgirtsou/Documents/June2019/datasets'
os.chdir('/home/sgirtsou/Documents/June2019/datasets')

