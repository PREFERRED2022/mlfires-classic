from csv import DictWriter
import os
import numpy as np
import pandas as pd
import hashlib
import re
from hyperopt import tpe, rand

def writemetrics(metrics, mean_metrics, hpresfile, allresfile):
    if not os.path.exists(os.path.dirname(hpresfile)):
        os.makedirs(os.path.dirname(hpresfile))
    writeheader = True if not os.path.isfile(hpresfile) else False
    with open(hpresfile, 'a') as _f:
        dw = DictWriter(_f, fieldnames=mean_metrics.keys(), quotechar='"')
        if writeheader:
            dw.writeheader()
        dw.writerow(mean_metrics)
    writeheader = True if not os.path.isfile(allresfile) else False
    with open(allresfile, 'a') as _f:
        dw = DictWriter(_f, fieldnames=metrics[0].keys(), quotechar='"')
        if writeheader:
            dw.writeheader()
        for m in metrics:
            dw.writerow(m)

def write_score(fname, id_pd, dates_pd, y_val, y_scores):
    if not os.path.exists(fname):
        if id_pd is None:
            pdscores = pd.Series(y_val).rename('fire').to_frame()
        else:
            pdscores = pd.concat([id_pd, dates_pd, y_val], axis=1)
            pdscores['id'].apply(np.int64)
    else:
        pdscores = pd.read_csv(fname, dtype={'id': str, 'firedate': str})
    col = str(len(pdscores.columns)) if id_pd is None else str(len(pdscores.columns)-2)
    y_sc_pd = pd.Series(y_scores)
    y_sc_pd.rename(col, inplace=True)
    score_pd = pd.concat([pdscores, y_sc_pd], axis=1)
    score_pd.to_csv(fname, index=False)
    pdscores = None

def gethashrow(row):
    #m = hashlib.sha256()
    m = hashlib.sha3_512()
    m.update(row.tobytes())
    return m.hexdigest()

def gethashdict(X):
    hashes = np.apply_along_axis(gethashrow, 1, X)
    Xhash = {}
    for idx, h in np.ndenumerate(hashes):
        Xhash[h] = idx[0]
    return Xhash

def updateYrows(Xval, Yval, Xhash, Yall):
    updhashes = np.apply_along_axis(gethashrow, 1, Xval)
    for idx, h in np.ndenumerate(updhashes):
        Yall[Xhash[h]] = Yval[idx[0]]

def get_filename(opt_target, modeltype, desc, aggr='mean', ext='.csv', resultsfolder='.'):
    base_name = os.path.join('hypres_'+ modeltype \
                             + '_' + desc + '_'+ aggr+'_'+\
                             "".join([ch for ch in opt_target if re.match(r'\w', ch)]) + '_')
    cnt = 1
    while os.path.exists(os.path.join(resultsfolder, '%s%d.csv' % (base_name, cnt))):
        cnt += 1
    fname = '%s%d%s' % (os.path.join(resultsfolder, base_name), cnt, ext)
    return fname

def get_hyperopt_algo(hypalgoparam):
    if hypalgoparam == 'tpe':
        hypalgo = tpe.suggest
    elif hypalgoparam == 'random':
        hypalgo = rand.suggest
    else:
        hypalgo = None
    return hypalgo