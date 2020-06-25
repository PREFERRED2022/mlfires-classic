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


def normalized_values(y,dfmax, dfmin, dfmean, dfstd, t = None):
    if not t:
        a = (y- dfmin) / (dfmax - dfmin)
        return(a)
    elif t=='std':
        a = (y - dfmean) / dfstd
        return(a)
    elif t=='no':
        return y

def normalize_dataset(X_unnorm_int, norm_type = None):
    X = DataFrame()
    for c in X_unnorm_int.columns:
        print(c)
        dfmax = X_unnorm_int[c].max()
        dfmin = X_unnorm_int[c].min()
        dfmean = X_unnorm_int[c].mean()
        dfstd = X_unnorm_int[c].std()
        X[c] = X_unnorm_int.apply(lambda x: normalized_values(x[c],dfmax, dfmin,dfmean,dfstd, norm_type),axis=1)
    return X

def convtoindex(y, lu_dict):
    return(lu_dict[y])

def indexdict(dfcol):
    lu = list(dfcol.unique())
    lu_dict = {x:lu.index(x)+1 for x in lu}
    return lu_dict

def index_string_values(X_unnorm, str_classes):
    indexdicts = {}
    for str_class in str_classes:
        indexdicts[str_class]=indexdict(X_unnorm[str_class])
    X_unnorm_int = X_unnorm.copy()
    for c in str_classes:
        print(c)
        X_unnorm_int[c] = X_unnorm.apply(lambda x: convtoindex(x[c],indexdicts[c]),axis=1)
    return X_unnorm_int

df = read_csv('/home/sgirtsou/Documents/ML-dataset_newLU/training_dataset.csv')
df = df.dropna()

df.loc[df.fire == 1, 'res_max'].std()
df.loc[(df.fire == 0) & (df.max_temp > 0.5), 'res_max'].std()

#df_greece_d = read_csv("/home/sgirtsou/Documents/ML-dataset_newLU/csvs_withfire/fire20190810.csv")
#df_greece = df_greece_d[df_greece_d.max_temp != '--']

#exclude from dataset fires of day
#firesofdaycells = df_greece[df_greece.fire ==1]['id'].tolist()
#daycellsindf = df1[df1['id'].isin(firesofdaycells) & df1['fire']==1]['id'].tolist()
#df = df[~df['id'].isin(firesofdaycells) & df['fire']==1]
#df = df[~df['id'].isin(firesofdaycells)]
#df.shape

#ff=df[df.max_temp_norm==min(df.max_temp_norm)]
#df_part = df[['id', 'strong', 'code_pi', 'max_temp',
#       'min_temp', 'mean_temp', 'res_max', 'dir_max', 'dom_vel', 'dom_dir',
#       'rain_7days', 'Near_dist', 'fclass', 'DEM', 'Slope', 'Curvature',
#       'Aspect', 'x', 'y', 'fire']].copy()
df_part = df[['id','max_temp','min_temp', 'mean_temp', 'res_max', 'dir_max', 'dom_vel', 'dom_dir', 'rain_7days','Corine',
       'Slope','DEM', 'Curvature','Aspect', 'ndvi','fire']].copy()

# split into input and output columns

#X, y_int = np.concatenate((df.values[1:, 3:14],df.values[1:, 15:19],df.values[1:, 20:22]),axis=1), df.values[1:, -3]
#X, y_int = df_part.values[:,:-1], df_part.values[:, -1]
#X_unnorm, y_int = df_part.values[:,:-1], df_part.values[:, -1]
X_unnorm, y_int = df_part[['max_temp','min_temp', 'mean_temp', 'res_max', 'dir_max', 'dom_vel', 'dom_dir', 'rain_7days',
                           'Corine','Slope','DEM', 'Curvature','Aspect', 'ndvi']], df_part['fire']
#n,id,firedate,lu_norm,fuel_norm,max_temp_norm,min_temp_norm,mean_temp_norm,res_max_norm,
#dir_max_norm,dom_vel_norm,dom_dir_norm,rain_7days_norm,Near_dist_norm,fclass_norm,Slope_norm,DEM_norm,
#Curvature_norm,Aspect_norm,fire,x,y

# ensure all data are floating point values
#X = X.astype('float32')

print(X_unnorm)

str_classes = ['Corine']
X_unnorm_int = index_string_values(X_unnorm, str_classes)
print(X_unnorm_int)

X = normalize_dataset(X_unnorm_int, 'std')
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
model.add(Dense(16, activation='relu', input_shape=(n_features,)))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
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
log_dir = os.path.join('.\\logs\\s2')
tb = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True, profile_batch = 100000000)
#model.fit(X_train[:,1:], y_train, epochs=100, batch_size=1000, callbacks = [es, tb])
model.fit(X_train, y_train, epochs=250, batch_size=1000, callbacks = [es, tb])

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

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for file in onlyfiles:
    if file.endswith('csv'):
        df_greece = pd.read_csv(file)
        print(file)
        #df_greece = read_csv('/home/sgirtsou/Documents/ML-dataset_newLU/csvs_withfire/fire20190810.csv')

#        df_greece = df_greece.rename(columns ={'Unnamed: 0': 'idx','DEM_500m_w': 'DEM', 'Aspect_500': 'Aspect', 'Curvat_500':'Curvature'})
#        print(df_greece.columns)

        df_greece = df_greece.dropna()

        #X_greece = df_greece[['max_temp',
        #       'min_temp', 'mean_temp', 'res_max', 'dir_max', 'dom_vel', 'dom_dir','DEM', 'Curvature','Aspect']]
        #       'Aspect', 'x', 'y', 'fire']].copy()
        X_greece_unnorm = df_greece[['max_temp', 'min_temp', 'mean_temp', 'res_max',
               'dir_max', 'dom_vel', 'dom_dir', 'rain_7days', 'Corine', 'DEM', 'Slope',
               'Curvature', 'Aspect', 'ndvi']]

        Y_greece = df_greece[['fire']]
        print('X_greece_unnorm.shape',X_greece_unnorm.shape)

        str_classes = []
        X_greeceunnorm_int = index_string_values(X_greece_unnorm, str_classes)
        #print(X_greeceunnorm_int)


        for c in X_greeceunnorm_int.columns:
            X_greeceunnorm_int[c] = pd.to_numeric(X_greeceunnorm_int[c], errors='coerce')

        X_greece = normalize_dataset(X_greeceunnorm_int, 'std')

        Y_pred_greece = model.predict(X_greece.values)

        Y_pred_greece.shape
        Y_pred_greece

        Y_pred = (Y_pred_greece[:,1]>0.5).astype(int)

        #(Y_pred_greece_f[:,1]>0.7).sum()

        Y_pred_greece_cl = model.predict_classes(X_greece.values)

        report = classification_report(Y_greece.values, Y_pred_greece_cl)
        #print(report)

        conf_matrix = confusion_matrix(Y_pred_greece_cl, Y_greece.values)
        f = open('/home/sgirtsou/Documents/June2019/confusion_matrix.csv', 'a')
        np.savetxt(f,  conf_matrix, delimiter=',',header= file[0:12], fmt='%f')
        f.close()
        #np.savetxt("cm_"+file[0:12]+".csv", conf_matrix, delimiter=",")
        #print(conf_matrix)
        '''
        fig = plt.figure()
        plt.matshow(conf_matrix)
        plt.title('Confusion Matrix_' + file[0:12])
        plt.colorbar()
        plt.ylabel('True Label')
        plt.xlabel('Predicated Label')
        plt.savefig('confusion_matrix'+ '.pdf')
        '''

        firerows = df_greece.loc[df_greece['fire']==1].index.tolist()
        nonfirerows = df_greece.loc[df_greece['fire']==0].index.tolist()

        Y_pred_greece_f = model.predict(X_greece.loc[firerows].values)

        Y_pred_f = (Y_pred_greece_f[:,1]>0.5).astype(int)

        #print(Y_pred_f)

        #print((Y_pred_greece_f[:,1]>0.9).sum())

        Y_pred_greece_nf = model.predict(X_greece.loc[nonfirerows].values)

        #print((Y_pred_greece_nf[:,1]>0.9).sum())

        Y_pred_greece_cl_df = DataFrame({'Class_pred': Y_pred_greece_cl})
        #print(Y_pred_greece_cl_df.shape)
        Y_pred_greece_df = DataFrame({'Class_0_proba': Y_pred_greece[:, 0], 'Class_1_proba': Y_pred_greece[:, 1]})
        #print(Y_pred_greece_df.shape)
        #print(df_greece.shape)

        df_results = pd.concat([df_greece, Y_pred_greece_cl_df, Y_pred_greece_df], axis=1)
        if file.startswith('no'):
            df_results.to_csv('/home/sgirtsou/Documents/June2019/NN_results/' + file[0:15] + '_res.csv')
        else:
            df_results.to_csv('/home/sgirtsou/Documents/June2019/NN_results/' + file[0:12] + '_res.csv')