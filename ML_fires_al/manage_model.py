from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
import tensorflow.keras.metrics
from sklearn.ensemble import RandomForestClassifier


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

    from tensorflow.keras.optimizers import Adam

    adam = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)

    if params['metric'] == 'accuracy':
        metrics = ['accuracy']
    elif params['metric'] == 'sparse':
        metrics = [tensorflow.metrics.SparseCategoricalAccuracy()]
    #elif params['metric'] == 'tn':
        #metrics = [tensorflow.metrics.TrueNegatives(),tensorflow.metrics.TruePositives()]
    model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=metrics)  # , AUC(multi_label=False)])

    return model

def create_RF_model(params):
    model = RandomForestClassifier(max_depth=params['max_depth'], n_estimators=params['n_estimators'])
    return model