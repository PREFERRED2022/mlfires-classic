import os
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from logitboost import LogitBoost
from sklearn.tree import DecisionTreeRegressor

#df=pd.read_csv("C:/Users/User/Documents/projects/FFP/ready dataset/dataset_norm_v2.csv")
df=pd.read_csv('/home/sgirtsou/Documents/ML-dataset_newLU/training_dataset.csv')

df = df.dropna() #drop null

print("df columns", df.columns)
print("df shape", df.shape)

#df_greece=pd.read_csv("/home/sgirtsou/Documents/ML-dataset_newLU/csvs_withfire/fire20190810.csv")

#print("df_greece cols",df_greece.columns)

#exclude from dataset fires of day
#firesofdaycells = df_greece[df_greece.fire ==1]['id'].tolist()
#daycellsindf = df1[df1['id'].isin(firesofdaycells) & df1['fire']==1]['id'].tolist()
#df = df[~df['id'].isin(firesofdaycells) & df['fire']==1]
#df = df1[~df1['id'].isin(firesofdaycells)]



# put feature columns 
#X= df[['strong','code_pi', 'max_temp',
#       'min_temp', 'mean_temp', 'res_max', 'dir_max', 'dom_vel', 'dom_dir','DEM', 'Curvature','Aspect', 'x', 'y']]

X= df[['max_temp','min_temp', 'mean_temp', 'res_max', 'dir_max', 'dom_vel', 'dom_dir', 'rain_7days',
                           'Corine','Slope','DEM', 'Curvature','Aspect', 'ndvi']]

#X= df[['DEM', 'max_temp', 'dom_dir', 'dir_max', 'res_max']]


# put output feature (e.g. mosquito population")
Y = df["fire"]

# training & testing data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.10) #20% hold out for testing

# initialize classifier
logitb = LogitBoost(DecisionTreeRegressor(max_depth=10),
                    n_estimators=100, random_state=0)

#train the model
logitb.fit(X_train, y_train)

# run on test set
Y_pred = logitb.predict(X_test)

report = classification_report(y_test, Y_pred)
print(report)



fi = pd.DataFrame({'feature': X_train.columns,
                   'importance': logitb.feature_importances_}).\
                    sort_values('importance', ascending = False)

importances = logitb.feature_importances_


indices = np.argsort(importances)[::-1]

print("Feature ranking:")


fi = pd.DataFrame({'feature': X.columns,
                   'importance': logitb.feature_importances_}).\
                    sort_values('importance', ascending = False)


l = [x for _,x in sorted(zip(logitb.feature_importances_,X.columns), reverse=True)]
for a,b in zip(sorted(logitb.feature_importances_, reverse=True), l):
    print("{0:12s}: {1}".format(b, a))
    
    
#df_greece=pd.read_csv("C:/Users/User/Documents/projects/FFP/one day 2019 dataset/ML-dataset_non_norm/data3_non_norm.csv")

#df_greece = df_greece.rename(columns ={'Unnamed: 0': 'idx','DEM_500m_w': 'DEM', 'Aspect_500': 'Aspect', 'Curvat_500':'Curvature'})
#df_greece = df_greece.drop(['fid','clcyp_vegd', 'g_id','id','geometry'], axis = 1)
#df_greece = df_greece.dropna().reset_index()
#df_greece = df_greece.drop(['index'], axis = 1)
#df_greece = df_greece.dropna()
#df_greece = df_greece[df_greece.max_temp != '--']

#df_greece = df_greece.loc[df_greece['fire']==1]
#X_greece = df_greece[['DEM', 'max_temp', 'dom_dir', 'dir_max', 'res_max']]

os.chdir('/home/sgirtsou/Documents/ML-dataset_newLU/csvs_withfire_results/NN_results')
mypath = '/home/sgirtsou/Documents/ML-dataset_newLU/csvs_withfire_results/NN_results'

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for file in onlyfiles:
    if file.endswith('csv'):
        df_greece = pd.read_csv(file)
        print(file)
        X_greece = df_greece[['max_temp', 'min_temp', 'mean_temp', 'res_max','dir_max', 'dom_vel', 'dom_dir', 'rain_7days',
                              'Corine', 'DEM', 'Slope','Curvature', 'Aspect', 'ndvi']]

        Y_greece = df_greece[['fire']]

        Y_pred_greece = logitb.predict(X_greece)
        report = classification_report(Y_greece, Y_pred_greece)
        print(report)

        conf_matrix = confusion_matrix(Y_pred_greece, Y_greece)
        f = open('/home/sgirtsou/Documents/ML-dataset_newLU/csvs_withfire_results/confusion_matrix.csv', 'a')
        np.savetxt(f, conf_matrix, delimiter=',', header=file[0:12]+'_lb', fmt='%f')
        f.close()
        print(conf_matrix)

        #for i in df_greece.loc[df_greece['fire']==1].index.tolist():
        #    print(logitb.predict_proba([X_greece.iloc[i]]))
        firerows = df_greece.loc[df_greece['fire']==1].index.tolist()
        nonfirerows = df_greece.loc[df_greece['fire']==0].index.tolist()
        fireprobas = logitb.predict_proba(X_greece.loc[firerows])
        nonfireprobas = logitb.predict_proba(X_greece.loc[nonfirerows])


        Y_pred_greece_proba = logitb.predict_proba(X_greece)

        Y_pred_greece_df = pd.DataFrame({'Class_pred_lb': Y_pred_greece})
        Y_pred_greece_proba_df = pd.DataFrame({'Class_0_proba_lb': Y_pred_greece_proba[:, 0], 'Class_1_proba_lb': Y_pred_greece_proba[:, 1]})
        df_results = pd.concat([df_greece, Y_pred_greece_df, Y_pred_greece_proba_df], axis=1)

        df_results.to_csv('/home/sgirtsou/Documents/ML-dataset_newLU/csvs_withfire_results/LB_results/' + file[0:12] + '_lb.csv')

        #fireprobahigh = fireprobas[:,1][fireprobas[:,1]>0.4]
        #print(fireprobahigh)
        #print(logitb.predict_proba(X_greece.iloc[df_greece.loc[df_greece['fire']==1].index.tolist()]))

