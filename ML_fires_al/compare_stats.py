import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import re
import matplotlib.pyplot as plt

def store_plot_stats(statspath, allstats, filesuf, plottitile):
    allstats.to_csv(os.path.join(statspath, 'stats_%s.csv' % filesuf))
    plotcols = [c for c in allstats.columns if 'prediction' in c or 'act' in c]
    plotstats = allstats[plotcols]
    #plt.rc('xtick', labelsize=14)
    ax1 = plotstats.plot.bar(figsize=(12, 8), fontsize=10, title=plottitile, grid=True)
    # ax1.legend(loc=5, fontsize=10)
    # ax1.set_ylabel('kw', fontdict={'fontsize': 24}, fontsize=16)
    # plt.show()

    plt.savefig(os.path.join(statspath, 'stats_%s.png' % filesuf))
    plt.close()

comp_path1='/home/sgirtsou/Documents/ML-dataset_newLU/csvs_withfire_results/stats_r4/stats_NN_LB_all.csv'
comp_path2='/home/sgirtsou/Documents/June2019/stats/r4/stats_june_2019.csv'
df_comp1 = pd.read_csv(comp_path1)
df_comp2 = pd.read_csv(comp_path2)

cl=[]
for c in df_comp1.columns:
    if 'Unnamed' in c:
        cl += [df_comp1[[c]][c].rename('Ranges')]
    elif not 'fire' in c and not 'event' in c:
        df_temp1 = df_comp1[[c]]
        df_temp2 = df_comp2[[c]]
        cl += [df_temp1[c].rename(c+' August'), df_temp2[c].rename(c + ' June')]

df_comb=pd.concat(cl, axis=1)
df_comb=df_comb.set_index('Ranges')

store_plot_stats(os.path.dirname(comp_path2), df_comb, 'compare_june_August', 'June August 2019 Comparison')

i=1