import pandas as pd
import fileutils

def retrieve_best_models(dir, filepattern, metrics, valst, testst, filters = []):
    best_models={}
    setfiles = [f for f in fileutils.find_files(dir, filepattern, listtype="walk")]
    df = None
    for f in setfiles:
        dftemp=pd.read_csv(f)
        df = dftemp if df is None else pd.concat([df,dftemp])
    for metric in metrics:
        df_flt = df
        for filt in filters:
            df_flt = df_flt[eval(filt)]
            '''
            if filt['operator']=='contains':
                df_flt = df_flt[df_flt[filt['column']].str.contains(filt['value'])]
            elif filt['operator']=='>':
                df_flt = df_flt[df_flt[filt['column']] > filt['value']]
            elif filt['operator']=='<':
                df_flt = df_flt[df_flt[filt['column']] < filt['value']]
            elif filt['operator']=='==':
                df_flt = df_flt[df_flt[filt['column']] == filt['value']]
            '''
        df_sorted = df_flt.sort_values(by=['%s %s'%(metric,valst)], ascending=False)
        #best_models['%s %s'%(metric,testst)] = [{'params':eval(df_sorted.iloc[0]['params']), 'trial':df_sorted.iloc[0]['trial']}]
        if 'trial' in df_sorted.columns:
            best_models['%s %s'%(metric,testst)] = [{'params':eval(df_sorted.iloc[0]['params']), 'trial':df_sorted.iloc[0]['trial']}]
        else:
            best_models['%s %s'%(metric,testst)] = [{'params':eval(df_sorted.iloc[0]['params']), 'trial':1}]
    return best_models
'''
metrics=['auc', 'hybrid2', 'hybrid5', 'NH2', 'NH5', 'NH10']
best_models=retrieve_best_models('/home/aapostolakis/Documents/ffpdata/results/aris/', '*2018only*', metrics, 'val.', 'test')
for m in metrics:
    print('%s test : %s'%(m,best_models['%s test'%(m)]))
'''
