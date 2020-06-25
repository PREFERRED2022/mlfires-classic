#!/usr/bin/env python

import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import re

os.chdir('/home/sgirtsou/Documents/June2019/Comb_results')
mypath = '/home/sgirtsou/Documents/June2019/Comb_results'


ranges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def set_percents(field, stats_field, allstats):
    sum = allstats[field].sum()
    allstats[stats_field] = allstats[field]/sum

def get_groups(name, fieldranges, conmbined_csv, ranges, filterfield=None, filter=None):
    if filterfield:
        combined_csv_filt = combined_csv[combined_csv[filterfield] == filter]
    else:
        combined_csv_filt = combined_csv
    combined_cut = pd.cut(combined_csv_filt[fieldranges], ranges)
    combined_cut = combined_cut.rename(name)
    combined_csv_r = pd.concat([combined_csv_filt, combined_cut], axis=1)
    groups = combined_csv_r.groupby(combined_cut).count()[fieldranges]
    return groups


groupfields = [{'field': 'Class_1_proba', 'name': 'NN'}, {'field': 'Class_1_proba_lb', 'name': 'LB'},
               {'field': 'Comb_proba_1', 'name': 'Ensemble'}]


# NN	fire_NN	event_NN
# LB	fire_LB	event_LB
# Ensemble	fire_Ensemble	event_Ensemble
# NN prediction	NN act. fire	NN act. event
# LB prediction	LB act. fire	LB act. event
# Ensemble prediction	Ensemble act. fire	Ensemble act. event

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) and 'comb' in f]

for f in onlyfiles:
    fdate = re.search('[0-9]{8}', f).group(0)
    combined_csv = pd.read_csv(f)
    allgroups = []
    for gf in groupfields:
        filt = gf['filter'] if 'filter' in gf else None
        g = get_groups(gf['name'], gf['field'], combined_csv, ranges, filt).rename(gf['name'])
        g_fire = get_groups(gf['name'], gf['field'], combined_csv, ranges, 'fire', 1).rename('fire_%s'%gf['name'])
        g_event = get_groups(gf['name'], gf['field'], combined_csv, ranges, 'fire_id', 1).rename('event_%s'%gf['name'])
        allgroups += [g, g_fire, g_event]

    allstats = pd.concat(allgroups, axis=1)

    for gf in groupfields:
        set_percents(gf['name'], '%s prediction' % gf['name'], allstats)
        set_percents('fire_%s' % gf['name'], '%s act. fire' % gf['name'], allstats)
        set_percents('event_%s' % gf['name'], '%s act. event' % gf['name'], allstats)

    statspath='/home/sgirtsou/Documents/June2019/stats/r%d'%(len(ranges)-1)
    if not os.path.exists(statspath):
        os.makedirs(statspath)
    #allstats.to_csv(os.path.join(statspath,'stats_%s.csv'%fdate))

    plotcols=[c for c in allstats.columns if 'prediction' in c or 'act' in c]
    plotstats=allstats[plotcols]
    #plotstats.plot.bar()
    allstats.plot.bar()
    i=1

