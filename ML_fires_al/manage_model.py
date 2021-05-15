from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
import tensorflow.keras.metrics
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from MLscores import calc_metrics, metrics_dict

def run_predict(model, modeltype, X):
    if modeltype == 'tensorflow':
        y_scores = model.predict(X)
    elif modeltype == 'sklearn':
        y_scores = model.predict_proba(X)
    predict_class = lambda p: int(round(p))
    predict_class_v = np.vectorize(predict_class)
    y_pred = predict_class_v(y_scores[:, 1])
    return y_scores, y_pred

def run_predict_and_metrics(model, modeltype, X, y, metricset, dontcalc=False, numaucthres=200, debug=True):
    if dontcalc:
        zerostuple = tuple([0]*16)
        return metrics_dict(*zerostuple, metricset)
    if debug:
        print("Running prediction...")
    y_scores, y_pred = run_predict(model, modeltype, X)
    metricsdict = metrics_dict(*calc_metrics(y, y_scores, y_pred, numaucthres=numaucthres, debug=debug), metricset)
    return metricsdict, y_scores

def create_model(modeltype, params, X_train):
    if modeltype == 'tensorflow':
        model = create_NN_model(params, X_train)
    elif modeltype == 'sklearn':
        model = create_sklearn_model(params, X_train)
    return model

def fit_model(modeltype, model, params, X_train, y_train, X_val, y_val):
    if modeltype == 'tensorflow':
        es = EarlyStopping(monitor=params['ES_monitor'], patience=params['ES_patience'], min_delta=params['ES_mindelta'])
        res = model.fit(X_train, y_train, batch_size=params['batch_size'], epochs=params['max_epochs'], verbose=0, callbacks=[es],
                        validation_data=(X_val, y_val), class_weight=params['class_weights'])
    elif modeltype == 'sklearn':
        res = model.fit(X_train, y_train)
    return model, res


def create_NN_model(params, X):
    # define model
    model = Sequential()
    n_features = X.shape[1]
    intlayers = int(params['n_internal_layers'][0])
    model.add(Dense(params['n_internal_layers'][1]['layer_1_' + str(intlayers) + '_nodes'], activation='relu',
                    input_shape=(n_features,)))
    if params['dropout']:
        model.add(Dropout(0.5))
    for i in range(2, intlayers + 2):
        model.add(Dense(int(params['n_internal_layers'][1]['layer_' + str(i) + '_' + str(intlayers) + '_nodes']),
                        activation='relu'))
        if params['dropout']:
            model.add(Dropout(0.5))

        # model.add(Dense(1, activation='sigmoid'))
    model.add(Dense(2, activation='softmax'))

    # compile the model

    if params['optimizer']['name']=='Adam':
        # adam = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        if params['optimizer']['adam_params'] is None:
            opt = Adam()
        else:
            opt = Adam(learning_rate=params['optimizer']['adam_params']['learning_rate_adam'], beta_1=params['optimizer']['adam_params']['beta_1'],
                       beta_2=params['optimizer']['adam_params']['beta_2'],amsgrad=params['optimizer']['adam_params']['amsgrad'])
    elif params['optimizer']['name']=='SGD':
        opt = SGD(learning_rate=params['optimizer']['learning_rate_SGD'])

    if params['metric'] == 'accuracy':
        metrics = ['accuracy']
    elif params['metric'] == 'sparse':
        metrics = [tensorflow.metrics.SparseCategoricalAccuracy()]
    #elif params['metric'] == 'tn':
        #metrics = [tensorflow.metrics.TrueNegatives(),tensorflow.metrics.TruePositives()]
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=metrics)  # , AUC(multi_label=False)])

    return model

def create_sklearn_model(params, X):
    n_features = X.shape[1]
    if params['algo']=='RF':
        max_feat = int(n_features/10*params['max_features'])
        model = RandomForestClassifier(max_depth=params['max_depth'], n_estimators=params['n_estimators'], min_samples_split=params['min_samples_split'], \
                                       min_samples_leaf=params['min_samples_leaf'],criterion=params['criterion'],max_features=max_feat,
                                       bootstrap=params['bootstrap'], class_weight=params['class_weights']
                                       )
    return model