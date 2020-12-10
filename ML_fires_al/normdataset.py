from pandas import DataFrame
import os
import json
import multiprocessing
import time

def normalized_values(y,dfmax, dfmin, dfmean, dfstd, t = None):
    if not t:
        a = (y- dfmin) / (dfmax - dfmin)
        return(a)
    elif t=='std':
        a = (y - dfmean) / dfstd
        return(a)
    elif t=='no':
        return y

def dataset_sanity_check(df):
    for c in [cl for cl in df.columns if 'bin' not in cl]:
        print('column %s - max: %s, min : %s, mean: %s, std: %s'%(c, df[c].max(), df[c].min(), df[c].mean(), df[c].std()))

def apply_norm(normdf, unnormdf, col, dfmax, dfmin, dfmean, dfstd, norm_type):
    #normdf = args[0]
    #unnormdf = args[1]
    #col = args[2]
    #dfmax = args[3]
    #dfmin = args[4]
    #dfmean = args[5]
    #dfstd = args[6]
    #norm_type = args[7]
    normdf[col] = unnormdf.apply(lambda x: normalized_values(x[col], dfmax, dfmin, dfmean, dfstd, norm_type), axis=1)

def normalize_dataset(df, norm_type = None, aggrfile = None):
    X = DataFrame()
    aggrs = None
    if aggrfile and os.path.exists(aggrfile):
        with open(aggrfile) as aggrf:
            aggrs = json.loads(aggrf.read())
    arglist = []
    plist = []
    for c in df.columns:
        print("Normalize column:%s"%c)
        if not 'bin' in c:
            if not aggrs is None:
                if not c in aggrs:
                    incol = [cl for cl in aggrs if cl.upper() in c.upper() or c.upper() in cl.upper()]
                    if len(incol) == 0:
                        print("Failed to find aggregatons for %s" % c)
                        continue
                    else:
                        c = incol[0]
                dfmax = aggrs[c]['max'] if 'max' in aggrs[c] else None
                dfmin = aggrs[c]['min'] if 'min' in aggrs[c] else None
                dfmean = aggrs[c]['mean'] if 'mean' in aggrs[c] else None
                dfstd = aggrs[c]['std'] if 'std' in aggrs[c] else None
            else:
                dfmax = df[c].max()
                dfmin = df[c].min()
                dfmean = df[c].mean()
                dfstd = df[c].std()
            X[c] = df.apply(lambda x: normalized_values(x[c], dfmax, dfmin, dfmean, dfstd, norm_type), axis=1)
            dataset_sanity_check(X[[c]])
            #arglist.append((X, df, c, dfmax, dfmin, dfmean, dfstd, norm_type))
            #p = multiprocessing.Process(target=apply_norm, args=(X, df, c, dfmax, dfmin, dfmean, dfstd, norm_type))
            #p.start()
            #plist.append(p)

        else:
            X[c] = df[c]
            print('binary are not normalized')

        #while(any(p.is_alive() for p in plist)):
        #    time.sleep(5)
        #for c in df.columns:
        #    dataset_sanity_check(X[[c]])

    return X


def convtoindex(y, lu_dict):
    return(lu_dict[y])


def indexdict(dfcol):
    lu = list(dfcol.unique())
    lu_dict = {x:lu.index(x)+1 for x in lu}
    return lu_dict


def index_string_values(X_unnorm, str_classes):
    indexdicts = {}
    for str_class in str_classes:
        indexdicts[str_class]=indexdict(X_unnorm[str_class])
    X_unnorm_int = X_unnorm.copy()
    for c in str_classes:
        print(c)
        X_unnorm_int[c] = X_unnorm.apply(lambda x: convtoindex(x[c],indexdicts[c]),axis=1)
    return X_unnorm_int
