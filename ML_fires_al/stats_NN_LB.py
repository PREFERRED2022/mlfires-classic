#!/usr/bin/env python

import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import re
import matplotlib.pyplot as plt


def set_percents(field, stats_field, allstats):
    sum = allstats[field].sum()
    allstats[stats_field] = allstats[field] / sum


def get_grouped(name, fieldranges, conmbined_csv, ranges, filterfield=None, filter=None):
    if filterfield and filter != 'notnull':
        combined_csv_filt = combined_csv[combined_csv[filterfield] == filter]
    elif filterfield and filter == 'notnull':
        combined_csv_filt = combined_csv[combined_csv[filterfield].notnull()]
        combined_csv_filt = combined_csv_filt.loc[combined_csv_filt.groupby([filterfield])[fieldranges].idxmax()]
    else:
        combined_csv_filt = combined_csv
    combined_cut = pd.cut(combined_csv_filt[fieldranges], ranges)
    combined_cut = combined_cut.rename(name)
    combined_csv_r = pd.concat([combined_csv_filt, combined_cut], axis=1)
    groups = combined_csv_r.groupby(combined_cut).count()[fieldranges]
    return groups


def calc_percents(allstats, groupfields):
    for gf in groupfields:
        set_percents(gf['name'], '%s prediction' % gf['name'], allstats)
        set_percents('fire_%s' % gf['name'], '%s act. fire' % gf['name'], allstats)
        set_percents('event_%s' % gf['name'], '%s act. event' % gf['name'], allstats)

def get_stats(combined_csv, groupfields, ranges):
    allgroups = []
    for gf in groupfields:
        filt = gf['filter'] if 'filter' in gf else None
        g = get_grouped(gf['name'], gf['field'], combined_csv, ranges, filt).rename(gf['name'])
        g_fire = get_grouped(gf['name'], gf['field'], combined_csv, ranges, 'fire', 1).rename('fire_%s' % gf['name'])
        g_event = get_grouped(gf['name'], gf['field'], combined_csv, ranges, 'fire_id', 'notnull').rename('event_%s' % gf['name'])
        allgroups += [g, g_fire, g_event]

    allstats = pd.concat(allgroups, axis=1)

    calc_percents(allstats, groupfields)

    return allstats

def store_plot_stats(allstats, filesuf, plottitile):
    allstats.to_csv(os.path.join(statspath, 'stats_%s.csv' % filesuf))
    plotcols = [c for c in allstats.columns if 'prediction' in c or 'act' in c]
    plotstats = allstats[plotcols]
    #plt.rc('xtick', labelsize=14)
    ax1 = plotstats.plot.bar(figsize=(12, 8), fontsize=10, title=plottitile)
    # ax1.legend(loc=5, fontsize=10)
    # ax1.set_ylabel('kw', fontdict={'fontsize': 24}, fontsize=16)
    # plt.show()

    plt.savefig(os.path.join(statspath, 'stats_%s.png' % filesuf))
    plt.close()


os.chdir('/home/sgirtsou/Documents/June2019/Comb_results')
mypath = '/home/sgirtsou/Documents/June2019/Comb_results'


ranges = [0.0, 0.25, 0.5, 0.75, 1.0] #4 ranges
#ranges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] #5 ranges
#ranges = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] #10 ranges

# NN	fire_NN	event_NN
# LB	fire_LB	event_LB
# Ensemble	fire_Ensemble	event_Ensemble
# NN prediction	NN act. fire	NN act. event
# LB prediction	LB act. fire	LB act. event
# Ensemble prediction	Ensemble act. fire	Ensemble act. event
groupfields = [{'field': 'Class_1_proba', 'name': 'NN'}, {'field': 'Class_1_proba_lb', 'name': 'LB'},
               {'field': 'Comb_proba_1', 'name': 'Ensemble'}]
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) and 'comb' in f] #and not 'no' in f]
monthstats=pd.DataFrame()
statspath = '/home/sgirtsou/Documents/June2019/stats/r%d' % (len(ranges) - 1)
if not os.path.exists(statspath):
    os.makedirs(statspath)

cnt=1
for f in onlyfiles:
    print('processing file %s (%d/%d)'%(f, cnt,len(onlyfiles)))
    cnt+=1
    fdate = re.search('[0-9]{8}', f).group(0)
    combined_csv = pd.read_csv(f)
    allstats = get_stats(combined_csv, groupfields, ranges)
    plottitle='%s/%s/%s' % (fdate[6:8], fdate[4:6], fdate[0:4])
    store_plot_stats(allstats, fdate, plottitle)
    monthstats = pd.concat([monthstats,allstats])

print('processing month file')
statcols = [c for c in monthstats.columns if not 'prediction' in c and not 'act' in c]
monthstats = monthstats[statcols]
sum_monthstats = monthstats.groupby(monthstats.index).sum()
calc_percents(sum_monthstats, groupfields)
store_plot_stats(sum_monthstats, 'june_2019', 'June 2019')
