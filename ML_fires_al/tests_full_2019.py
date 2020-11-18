from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

df = pd.read_csv('/home/sgirtsou/Documents/ML-dataset_newLU/dataset_corine_level2_onehotenc.csv')


data = df[['id','max_temp', 'min_temp', 'mean_temp', 'res_max', 'dom_vel', 'rain_7days', 'dem', 'slope',
       'curvature', 'aspect','ndvi_new', 'evi','dir_max_1','dir_max_2', 'dir_max_3', 'dir_max_4', 'dir_max_5', 'dir_max_6',
       'dir_max_7', 'dir_max_8', 'dom_dir_1', 'dom_dir_2', 'dom_dir_3','dom_dir_4', 'dom_dir_5', 'dom_dir_6', 'dom_dir_7', 'dom_dir_8',
       'corine_2_11', 'corine_2_12', 'corine_2_13', 'corine_2_14','corine_2_21', 'corine_2_22', 'corine_2_23', 'corine_2_24',
       'corine_2_31', 'corine_2_32', 'corine_2_33', 'corine_2_41','corine_2_42', 'corine_2_51', 'corine_2_52','fire']].copy()

data = data.dropna()

X = data[['max_temp', 'min_temp', 'mean_temp', 'res_max', 'dom_vel', 'rain_7days', 'dem', 'slope',
       'curvature', 'aspect','ndvi_new', 'evi','dir_max_1','dir_max_2', 'dir_max_3', 'dir_max_4', 'dir_max_5', 'dir_max_6',
       'dir_max_7', 'dir_max_8', 'dom_dir_1', 'dom_dir_2', 'dom_dir_3','dom_dir_4', 'dom_dir_5', 'dom_dir_6', 'dom_dir_7', 'dom_dir_8',
       'corine_2_11', 'corine_2_12', 'corine_2_13', 'corine_2_14','corine_2_21', 'corine_2_22', 'corine_2_23', 'corine_2_24',
       'corine_2_31', 'corine_2_32', 'corine_2_33', 'corine_2_41','corine_2_42', 'corine_2_51', 'corine_2_52']]
Y = data['fire']

def random_forest(X,Y,test_set_x,test_set_y,file):
    chunksize = 10 ** 6
    parameters = {'max_rec':{'n_estimators': 1400, 'min_samples_split': 600, 'min_samples_leaf': 100, 'max_features': 1, 'max_depth': 200, 'criterion': 'gini', 'class_weight': {0: 2, 1: 8}, 'bootstrap': False},
                  'max_fscore':{'n_estimators': 250, 'min_samples_split': 180, 'min_samples_leaf': 100, 'max_features': 9, 'max_depth': 1000, 'criterion': 'gini', 'class_weight': {0: 3, 1: 7}, 'bootstrap': True},
                  'max_AUC': {'n_estimators': 170, 'min_samples_split': 120, 'min_samples_leaf': 50, 'max_features': 7, 'max_depth': 2000, 'criterion': 'entropy', 'class_weight': {0: 4, 1: 6}, 'bootstrap': True}}
    for key in parameters.keys():
        rf = RandomForestClassifier(**parameters[key])
        rf.fit(X, Y)
        for i,chunk in enumerate(pd.read_csv(file, chunksize=chunksize)):
            prediction = rf.predict(chunk)
            prediction_proba = rf.predict_proba(test_set_x)
            Y_predict = pd.DataFrame({'rf_pr':prediction})
            report = classification_report(test_set_y, Y_predict)
            Y_predict_proba = pd.DataFrame({'rf_class_0': prediction_proba[:,0],'rf_class_1': prediction_proba[:,1]})
            rf_prediction = pd.concat(test_set_x,test_set_y,Y_predict,Y_predict_proba)
        return(report,Y_predict_proba,rf_prediction)

def xgboost(X,Y,test_set):
    parameters = {'max_rec':{'subsample':0.8,'scale_pos_weight':101,'n_estimators':10,'max_depth':4,'lambda':21,'gamma':1000,'alpha':40},
                  'max_fscore':{'subsample':0.8,'scale_pos_weight':101,'n_estimators':10,'max_depth':4,'lambda':21,'gamma':1000,'alpha':40},
                  'max_prec': {'subsample':0.8,'scale_pos_weight':101,'n_estimators':10,'max_depth':4,'lambda':21,'gamma':1000,'alpha':40},
                  'max_AUC': {'subsample':0.8,'scale_pos_weight':101,'n_estimators':10,'max_depth':4,'lambda':21,'gamma':1000,'alpha':40}}
    for key in parameters.keys():
        xg = XGBClassifier(**parameters[key])
        xg.fit(X, Y)
        prediction = xg.predict(test_set)
        prediction_proba = xg.predict_proba(test_set)
        Y_predict = pd.DataFrame({'xg_pr':prediction})
        Y_predict_proba = pd.DataFrame({'xg_class_0': prediction_proba[:,0],'xg_class_1': prediction_proba[:,1]})

def extra_trees(X,Y,test_set):
    parameters = {'max_rec':{'oob_score': False, 'n_estimators': 800, 'min_samples_split': 100, 'min_samples_leaf': 5, 'max_features': 4, 'max_depth': 6, 'criterion': 'gini', 'class_weight': {0: 2, 1: 8}, 'bootstrap': True},
                  'max_fscore':{'oob_score': False, 'n_estimators': 1000, 'min_samples_split': 400, 'min_samples_leaf': 5, 'max_features': 41, 'max_depth': 4, 'criterion': 'entropy', 'class_weight': {0: 4, 1: 6}, 'bootstrap': True},
                  'max_prec': {'oob_score': True, 'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 10, 'max_features': 19, 'max_depth': 32, 'criterion': 'entropy', 'class_weight': {0: 4, 1: 6}, 'bootstrap': True}}
    for key in parameters.keys():
        et = ExtraTreesClassifier(**parameters[key])
        et.fit(X, Y)
        prediction = et.predict(test_set)
        prediction_proba = et.predict_proba(test_set)
        Y_predict = pd.DataFrame({'et_pr':prediction})
        Y_predict_proba = pd.DataFrame({'et_class_0': prediction_proba[:,0],'et_class_1': prediction_proba[:,1]})
#        rf_prediction = pd.concat(test_set,Y_predict,Y_predict_proba)

training_set = pd.read_csv('/home/sgirtsou/Documents/ML-dataset_newLU/dataset_corine_level2_onehotenc.csv')
test_set = '/home/sgirtsou/Documents/test_datasets_19/test_datasets_2019_dummies/june_2019_dataset_fire_sh_dummies.csv'

features = ['max_temp', 'min_temp', 'mean_temp', 'res_max', 'dom_vel', 'rain_7days', 'dem', 'slope',
       'curvature', 'aspect','ndvi_new', 'evi','dir_max_1','dir_max_2', 'dir_max_3', 'dir_max_4', 'dir_max_5', 'dir_max_6',
       'dir_max_7', 'dir_max_8', 'dom_dir_1', 'dom_dir_2', 'dom_dir_3','dom_dir_4', 'dom_dir_5', 'dom_dir_6', 'dom_dir_7', 'dom_dir_8',
       'corine_2_11', 'corine_2_12', 'corine_2_13', 'corine_2_14','corine_2_21', 'corine_2_22', 'corine_2_23', 'corine_2_24',
       'corine_2_31', 'corine_2_32', 'corine_2_33', 'corine_2_41','corine_2_42', 'corine_2_51', 'corine_2_52']
fire = ['fire']

training_set_X = training_set[features]
test_set_X = test_set[features]

training_set_Y = training_set[fire]
test_set_Y = test_set[fire]


report,Y_predict_proba,rf_prediction = random_forest(training_set_X,training_set_Y,test_set_X,test_set_Y,test_set)

