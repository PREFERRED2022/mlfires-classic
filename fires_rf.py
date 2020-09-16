import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report




#df=pd.read_csv("/home/sgirtsou/Documents/ML-dataset_newLU/training_dataset.csv")
df_x = pd.read_csv("/home/sgirtsou/Documents/ML-dataset_newLU/train.csv")
test = pd.read_csv("/home/sgirtsou/Documents/ML-dataset_newLU/test.csv")

df_x = df_x.dropna() #drop null
test = test.dropna()

X= df_x[['max_temp','min_temp', 'mean_temp', 'res_max', 'dir_max', 'dom_vel', 'dom_dir', 'rain_7days',
                           'Corine','Slope','DEM', 'Curvature','Aspect', 'ndvi']]

#X= df[['DEM', 'max_temp', 'dom_dir', 'dir_max', 'res_max']]


# put output feature (e.g. mosquito population")
Y = df_x["fire"]

test_x = test[['max_temp','min_temp', 'mean_temp', 'res_max', 'dir_max', 'dom_vel', 'dom_dir', 'rain_7days',
                           'Corine','Slope','DEM', 'Curvature','Aspect', 'ndvi']]
test_y = test["fire"]

# training & testing data
#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.10) #10% hold out for testing

# initialize classifier
rf = RandomForestClassifier(n_estimators=325, min_samples_split=200, oob_score=True, bootstrap=True, criterion= 'entropy')

#train the model
#rf.fit(X_train, y_train)
rf.fit(X, Y)
# run on test set
#Y_pred = rf.predict(X_test)
Y_pred = rf.predict(test_x)

#report = classification_report(y_test, Y_pred)
report = classification_report(test_y, Y_pred)
#print(report)



fi = pd.DataFrame({'feature': X_train.columns,
                   'importance': rf.feature_importances_}).\
                    sort_values('importance', ascending = False)

importances = rf.feature_importances_


indices = np.argsort(importances)[::-1]


fi = pd.DataFrame({'feature': X.columns,
                   'importance': rf.feature_importances_}).\
                    sort_values('importance', ascending = False)


l = [x for _,x in sorted(zip(rf.feature_importances_,X.columns), reverse=True)]
for a,b in zip(sorted(rf.feature_importances_, reverse=True), l):
    print("{0:12s}: {1}".format(b, a))
