from csv import DictWriter
import os
import numpy as np
import pandas as pd
import hashlib

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

def write_score(fname, iddatedf, y_scores, colname):
    y_pd = pd.Series(y_scores)
    y_pd.rename(colname, inplace=True)
    score_pd = pd.concat([iddatedf, y_pd], axis=1)
    score_pd.to_csv(fname, index=False)

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


