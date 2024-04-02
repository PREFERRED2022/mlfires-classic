import sys
sys.path.append("/mnt/nvme2tb/ffp/code/mlfires/ML_fires_al/")
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Flatten, Concatenate, concatenate, Dropout, Lambda
from keras.models import Model
from keras.layers import Embedding
from tqdm import tqdm
import shap
from manage_model import create_NN_model, create_sklearn_model, allowgrowthgpus, mm_load_model
import os
import pandas as pd
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from MLscores import calc_metrics, metrics_dict, cmvals, recall, hybridrecall
from functools import partial
from check_and_prepare_dataset import load_dataset, prepare_dataset
import pathlib
from best_models import retrieve_models_by_id
import xarray as xr
import numpy as np
import os
import re
import geopandas as gpd
from crop_dataset_files_op177 import cropfile
from functools import partial
import dice_ml
from dice_ml import Dice
from datetime import datetime

def applyid(df):
    df['xposst'] = (df['x'] * 10000).apply('{:06.0f}'.format)
    df['yposst'] = (df['y'] * 10000).apply('{:06.0f}'.format)
    df['id'] = df['xposst']+df['yposst']
    df.drop(columns=['xposst', 'yposst'],inplace=True)
    return df

#get coarse resolution dataset for local explainability

# Load dataset with parameters of specific model
def xai_ds_load(coarsefile):
    data_dir = "/mnt/nvme2tb/ffp/datasets"
    models_dir = "/mnt/nvme2tb/ffp/results/bestmodels"
    xaidir='/mnt/nvme2tb/ffp/datasets/xai'

    filters = ["df_flt['params'].str.contains(\"'dropout': None\")"]
    opt_targets = [
        "auc", "f1-score 1", "hybrid1", "hybrid2", "hybrid5", "NH2", "NH5", "NH10"
    ]
    testfpattern = cvrespattern = "*NN_ns*mean*"
    modelparams = retrieve_models_by_id(785, "hybrid2 test", models_dir, testfpattern,
                                      opt_targets, "val.", "test", filters, 3)

    x_t, y_t, _ = load_dataset(coarsefile, modelparams["params"]["feature_drop"])
    x_t = x_t.reindex(sorted(x_t.columns), axis=1)
    dataset = pd.concat([x_t, y_t], axis=1)
    #dataset = dataset.sample(frac=1).reset_index(drop=True)
    #dataset
    return dataset, modelparams

# Load model file
def xai_model_load(modelparams):
    models_dir = "/mnt/nvme2tb/ffp/results/bestmodels"
    model_file = "hypres_tf_ns_ncv_do_2019_model_id_785_r_0_hybrid2test_1.h5"
    #weights_file = "hypres_tf_ns_ncv_do_2019_weights_id_785_r_0_hybrid2test_1.cpkt"
    model = mm_load_model(pathlib.Path(models_dir, "entiremodels", model_file),
                          "tf", modelparams["params"])
    #model.load_weights(pathlib.Path(models_dir, "weights", weights_file))
    model.summary()
    return model


# create json from sorted class importance
# input:
#    xaimethod : e.g. shap, CF
#    unnormdf: unnormalized values dataset with id produced from unique x,y
#    xyid: x, y id (id produced from x,y)
#    invrendict: map dict with feature names in unnormalized values dataset
#    tup: tuple with feature name for json and value
# output:
#    json with feature name, importance value and unnormalized value

def create_json_xai(xaimethod, unnormdf, xyid, invrendict, tup):
    xyrow = unnormdf.loc[unnormdf['id'] == xyid]

    # choose right column names from unnormalized dataset
    if tup[0] in invrendict:
        scol = invrendict[tup[0]][4:]
    else:
        scol = tup[0]
    if tup[0] == 'xpos': scol = 'x'
    if tup[0] == 'ypos': scol = 'y'

    unnormval = xyrow[scol].item()
    jsonst = '{"feature":"%s", "%s" : %.2f, "value": %.2f}' % (tup[0], xaimethod, tup[1], unnormval)
    return jsonst


# create dataset with best n features sorted by importance for each instance
# input:
#    dfxai : dataset with columns feature names and rows the importances for each feature
#    numimp: number of best features to select
#    xaimethod : e.g. shap, CF
#    unnormdf: unnormalized values dataset with id produced from unique x,y
#    sorttype='abs' for only positive importance, 'both' for positive and negative
# output:
#    dataset with best "numimp" features sorted by importance for each instance

def sorted_imp(dfxai, numimp, xaimethod, unnormdf, sorttype='abs'):
    rendict = {c: 'cor_%s' % (re.search('\d+', c).group(0)) for c in dfxai.columns if c.startswith('bin_corine')}
    invrendict = {v: k for k, v in rendict.items()}
    dictxai = dfxai.rename(columns=rendict).drop(columns=['y', 'x']).to_dict('records')
    sortedimp = []
    for d in dictxai:
        if 'id' in d: xyid = d.pop('id')
        create_json_x = partial(create_json_xai, xaimethod, unnormdf, xyid, invrendict)
        if sorttype == 'abs':
            _sortimp = sorted(d.items(), key=lambda x: abs(x[1]), reverse=True)
            # sortimp=[_sortimp[0][1], _sortimp[1][1]]+list(map(create_json_x, _sortimp[2:2+numimp]))
            sortimp = list(map(create_json_x, _sortimp[0:numimp]))
        elif sorttype == 'both':
            _sortimp = sorted(d.items(), key=lambda x: x[1], reverse=True)
            # sortimpp = [_sortimp[0][1], _sortimp[1][1]]+list(map(create_json_x, _sortimp[2:2+numimp]))
            sortimpp = list(map(create_json_x, _sortimp[0:numimp]))
            _sortimp = sorted(d.items(), key=lambda x: x[1])
            sortimpn = list(map(create_json_x, _sortimp[0:numimp]))
            sortimp = sortimpp + sortimpn
        else:
            print('wrong sort type')
            return
        sortedimp += [sortimp]
    if sorttype == 'abs':
        cols = ['imp_%d' % i for i in list(range(1, numimp + 1))]
    elif sorttype == 'both':
        cols = ['imp_pos_%d' % i for i in list(range(1, numimp + 1))] + \
               ['imp_neg_%d' % i for i in list(range(1, numimp + 1))]
    # dfxaifin=pd.DataFrame(sortedimp, columns=['y','x']+cols)
    dfxaifin = pd.DataFrame(sortedimp, columns=cols)
    dfxaifin = pd.concat([dfxai[['y', 'x']], dfxaifin], axis=1)
    return dfxaifin


# create dataset with importance values
# input:
#    imp_values : importance values array
#    columns: Dataset colmns
#    coarsefile : coarse resolution file for getting instance id
# output:
#    dataset with importance values and id and x, y
def xai_xy_ds(imp_values, columns, coarsefile):
    coarsedf = pd.read_csv(coarsefile, dtype={'id': str})
    dfxai = pd.DataFrame(imp_values, columns=list(columns))
    dfxai = pd.concat([dfxai, coarsedf['id']], axis=1)
    # .to_csv('/mnt/nvme2tb/ffp/datasets/xai/20190929_shap_xai.csv', index=False)
    dfxai['x'] = dfxai['id'].str.slice(0, 6).astype(int) / 10000
    dfxai['y'] = dfxai['id'].str.slice(6, 12).astype(int) / 10000
    return dfxai

def fshap1(model, X):
    return model.predict(X, verbose=0)

def getunnorm(date):
    csvunnorm='/mnt/nvme2tb/ffp/datasets/prod/%s/%s.csv'%(date,date)
    dfunorm=pd.read_csv(csvunnorm)
    dfunormid=applyid(dfunorm)
    return dfunormid

def explain_day(date, ext=''):
    xaifolder = '/mnt/nvme2tb/ffp/datasets/xai/%s'%date
    prodfolder = '/mnt/nvme2tb/ffp/datasets/prod/%s'%date
    coarsefile = os.path.join(xaifolder, '%s_xai_%sinp.csv'%(date,ext))
    xaifile = os.path.join(xaifolder, '%s_xai_%sallfeat.csv'%(date,ext))
    predfile = os.path.join(prodfolder, '%s_pred_greece.csv'%date)


    #ypred = model.predict(X)
    #dfypred = pd.Series(ypred[:, 1]).rename('ypred')
    if not os.path.exists(xaifile):
        dfcoarsenorm, modparams = xai_ds_load(coarsefile)
        X = dfcoarsenorm.iloc[:, :-1]
        model = xai_model_load(modparams)
        fshap1p=partial(fshap1,model)
        explainer1 = shap.KernelExplainer(fshap1p, shap.kmeans(X, 20))
        shap_values = explainer1.shap_values(X, nsamples=150)
        # get df for explainability values
        dfxai = xai_xy_ds(shap_values[1], X.columns, coarsefile)
        dfxai.to_csv(xaifile, index=False)
    else:
        print('XAI csv file %s exists. Producing only sorted features and shp file'%xaifile)
        dfxai=pd.read_csv(xaifile, dtype={'id': str})
    dfunormid = getunnorm(date)
    dfxaisorted = sorted_imp(dfxai, 5, 'shap', dfunormid, sorttype='both')
    #dfxaisorted = pd.concat([dfypred, dfxaisorted], axis=1)
    dfxaisorted['stdtime'] = datetime.strptime(date, "%Y%m%d").strftime("%Y-%m-%d")
    dfxaisorted = applyid(dfxaisorted)
    dfpred = pd.read_csv(predfile, dtype={'id': str, 'risk': np.int16})
    dfxaisorted = pd.merge(dfpred, dfxaisorted, on='id', suffixes=('', '_xai')).\
        drop(columns=['y_xai', 'x_xai', 'ypred0']).rename(columns={'ypred1': 'ypred'})
    xaisortedfile=os.path.join(xaifolder, '%s_shap_%sxai.shp'%(date,ext))
    print('Convert csv to shp points %s'%xaisortedfile)
    gdf = gpd.GeoDataFrame(dfxaisorted, geometry=gpd.points_from_xy(dfxaisorted['x'], dfxaisorted['y'], z=None, crs=4326))
    #gdf = gpd.GeoDataFrame(dfxai, geometry=gpd.points_from_xy(dfxai['x'], dfxai['y'], z=None, crs=4326))
    gdf.drop(columns=['x', 'y'], inplace=True)
    gdf.to_file(xaisortedfile)


def main():
    args = sys.argv[1:]
    if len(args)==1:
        date=args[0]
        explain_day(date, 'ext_')

if __name__ == "__main__":
    main()