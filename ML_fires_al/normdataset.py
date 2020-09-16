from pandas import DataFrame


def normalized_values(y,dfmax, dfmin, dfmean, dfstd, t = None):
    if not t:
        a = (y- dfmin) / (dfmax - dfmin)
        return(a)
    elif t=='std':
        a = (y - dfmean) / dfstd
        return(a)
    elif t=='no':
        return y


def normalize_dataset(X_unnorm_int, norm_type = None):
    X = DataFrame()
    for c in X_unnorm_int.columns:
        print(c)
        if not 'bin' in c:
            dfmax = X_unnorm_int[c].max()
            dfmin = X_unnorm_int[c].min()
            dfmean = X_unnorm_int[c].mean()
            dfstd = X_unnorm_int[c].std()
            X[c] = X_unnorm_int.apply(lambda x: normalized_values(x[c],dfmax, dfmin,dfmean,dfstd, norm_type),axis=1)
        else:
            print('binary are not normalized')
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
