from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

data = pd.read_csv('/home/sgirtsou/Documents/ML-dataset_newLU/dataset_dummies.csv')
data = data.dropna()

X = data[['max_temp', 'min_temp', 'mean_temp',
       'res_max', 'dom_vel', 'rain_7days', 'Near_dist', 'DEM', 'Slope',
       'Curvature', 'Aspect', 'ndvi_new', 'evi', 'dir_max_1',
       'dir_max_2', 'dir_max_3', 'dir_max_4', 'dir_max_5', 'dir_max_6',
       'dir_max_7', 'dir_max_8', 'dom_dir_1', 'dom_dir_2', 'dom_dir_3',
       'dom_dir_4', 'dom_dir_5', 'dom_dir_6', 'dom_dir_7', 'dom_dir_8',
       'fclass_bridleway', 'fclass_footway', 'fclass_living_street',
       'fclass_motorway', 'fclass_path', 'fclass_pedestrian', 'fclass_primary',
       'fclass_residential', 'fclass_secondary', 'fclass_service',
       'fclass_steps', 'fclass_tertiary', 'fclass_track',
       'fclass_track_grade1', 'fclass_track_grade2', 'fclass_track_grade3',
       'fclass_track_grade4', 'fclass_track_grade5', 'fclass_trunk',
       'fclass_unclassified', 'fclass_unknown', 'Corine_111', 'Corine_112',
       'Corine_121', 'Corine_122', 'Corine_123', 'Corine_131', 'Corine_132',
       'Corine_133', 'Corine_142', 'Corine_211', 'Corine_212', 'Corine_213',
       'Corine_221', 'Corine_222', 'Corine_223', 'Corine_231', 'Corine_241',
       'Corine_242', 'Corine_243', 'Corine_311', 'Corine_312', 'Corine_313',
       'Corine_321', 'Corine_322', 'Corine_323', 'Corine_324', 'Corine_331',
       'Corine_332', 'Corine_333', 'Corine_334', 'Corine_411', 'Corine_421',
       'Corine_511', 'Corine_512', 'Corine_523']]
Y = data['fire']

def random_forest(X,Y,test_set):
    parameters = {'max_rec':{'n_estimators':1500,'max_features':1,'max_depth':10,'criterion':'gini','class_weight':{0:3,1:7},'bootstrap':True},
                  'max_fscore':{'n_estimators':250,'max_features':1,'max_depth':10,'criterion':'gini','class_weight':{0:4,1:6},'bootstrap':False},
                  'max_prec': {'n_estimators': 200, 'max_features': 3, 'max_depth': 1200, 'criterion': 'entropy','class_weight': {0: 3, 1: 7}, 'bootstrap': False},
                  'max_AUC': {'n_estimators': 1500, 'max_features': 4, 'max_depth': 6, 'criterion': 'entropy','class_weight': {0: 3, 1: 7}, 'bootstrap': False}}
    for key in parameters.keys():
        rf = RandomForestClassifier(**parameters[key])
        rf.fit(X, Y)
        prediction = rf.predict(test_set)
        prediction_proba = rf.predict_proba(test_set)
        Y_predict = pd.DataFrame({'rf_pr':prediction})
        Y_predict_proba = pd.DataFrame({'rf_class_0': prediction_proba[:,0],'rf_class_1': prediction_proba[:,1]})
#        rf_prediction = pd.concat(test_set,Y_predict,Y_predict_proba)

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