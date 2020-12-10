from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

df = pd.read_csv('/home/sgirtsou/Documents/perifereia/13082013_final.csv')

data = df[['id','dem_tif', 'lu_tif', 'aspect_tif', 'slope_tif', 'fire', 'firedate', 'x', 'y', 'max_temp',
       'min_temp', 'mean_temp', 'res_max', 'dir_max', 'dom_vel', 'dom_dir','ndvi']].copy()

X,Y = data[['dem_tif', 'lu_tif', 'aspect_tif', 'slope_tif','max_temp',
       'min_temp', 'mean_temp','res_max', 'dir_max', 'dom_vel', 'dom_dir','ndvi']], data['fire']

def random_forest(training_set_X,training_set_Y,test_set_X,test_set_Y,test_set):
    parameters = {'max_fscore':{'n_estimators': 1400, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 2, 'max_depth': 10, 'criterion': 'entropy', 'class_weight': {0: 1, 1: 10}, 'bootstrap': False},
                  'max_rec':{'n_estimators': 1500, 'min_samples_split': 1000, 'min_samples_leaf': 120, 'max_features': 5, 'max_depth': 500, 'criterion': 'entropy', 'class_weight': {0: 1, 1: 100}, 'bootstrap': True},
                  'max_AUC': {'n_estimators': 150, 'min_samples_split': 50, 'min_samples_leaf': 10, 'max_features': 8, 'max_depth': 200, 'criterion': 'entropy', 'class_weight': {0: 1, 1: 200}, 'bootstrap': False}}
    for key in parameters.keys():
        rf = RandomForestClassifier(**parameters[key])
        rf.fit(X, Y)
        prediction = rf.predict(test_set_X)
        prediction_proba = rf.predict_proba(test_set_X)
        Y_predict = pd.DataFrame({'rf_pr':prediction})
        report = classification_report(test_set_Y, Y_predict)
        Y_predict_proba = pd.DataFrame({'rf_class_0': prediction_proba[:,0],'rf_class_1': prediction_proba[:,1]})
        rf_prediction = pd.concat([test_set_X,test_set_Y,Y_predict,Y_predict_proba])
        rf_prediction.to_csv('/home/sgirtsou/Documents/perifereia/'+key+'.csv')
    return(report,Y_predict_proba,rf_prediction)

training_set = pd.read_csv('/home/sgirtsou/Documents/perifereia/13082013_final.csv')
test_set = pd.read_csv('/home/sgirtsou/Documents/perifereia/17082011_testset.csv')

training_set = training_set.dropna()
test_set = test_set.dropna()

features = ['dem_tif', 'lu_tif', 'aspect_tif', 'slope_tif','max_temp',
       'min_temp', 'mean_temp','res_max', 'dir_max', 'dom_vel', 'dom_dir','ndvi']

fire = ['fire']
training_set_X = training_set[features]
test_set_X = test_set[features]

training_set_Y = training_set[fire]
test_set_Y = test_set[fire]


report,Y_predict_proba,rf_prediction = random_forest(training_set_X,training_set_Y,test_set_X,test_set_Y,test_set)


i=1