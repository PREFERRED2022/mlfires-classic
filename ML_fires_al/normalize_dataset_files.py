import normdataset
import pandas as pd
import os

def normalizeset(lfiles):
    for f in lfiles:
        if f[-8:]!='norm.csv':
            df=pd.read_csv(f)
            print('before drop %d',len(df))
            df=df.dropna()
            print('after drop %d', len(df))
            normdf = normdataset.normalize_dataset(df,aggrfile='/data2/ffp/datasets/norm_values_ref_fin.json')
            fnorm=os.path.join(os.path.dirname(f), os.path.basename(f).split('.')[0]+'_norm.csv')
            normdf.to_csv(fnorm,index=False)

normalizeset(['/data2/ffp/datasets/trainingsets/newfull/traindataset_new.csv'])