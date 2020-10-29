import pandas as pd
from datetime import datetime
from datetime import timedelta
import os
import matplotlib.pyplot as plt

def get_grouped(name, fieldranges, combined_csv, ranges, filterfield=None, filter=None):
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

def set_percents(field, stats_field, allstats):
    sum = allstats[field].sum()
    allstats[stats_field] = allstats[field] / sum

def calc_percents(allstats, groupfields):
    for gf in groupfields:
        set_percents(gf['name'], '%s prediction' % gf['name'], allstats)
        set_percents('fire_%s' % gf['name'], '%s act. fire' % gf['name'], allstats)
        #set_percents('event_%s' % gf['name'], '%s act. event' % gf['name'], allstats)

def get_stats(combined_csv, groupfields, ranges):
    allgroups = []
    for gf in groupfields:
        filt = gf['filter'] if 'filter' in gf else None
        g = get_grouped(gf['name'], gf['field'], combined_csv, ranges, filt).rename(gf['name'])
        #g_fire = get_grouped(gf['name'], gf['field'], combined_csv, ranges, 'fire', 1).rename('fire_%s' % gf['name'])
        g_fire = get_grouped(gf['name'], gf['field'], combined_csv, ranges,'Date', 'notnull').rename('fire_%s' % gf['name'])

        allgroups += [g, g_fire]

    allstats = pd.concat(allgroups, axis=1)

    calc_percents(allstats, groupfields)

    return allstats

def store_plot_stats(statspath, allstats, filesuf, plottitile):
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

statsrangelist = [{'range':[0.0, 0.25, 0.5, 0.75, 1.0], 'allstats':pd.DataFrame() }, #4 ranges
                  {'range':[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], 'allstats':pd.DataFrame()}, #5 ranges
                  {'range':[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 'allstats':pd.DataFrame()} #10 ranges
]
groupfields = [{'field': 'Class_1_proba_nn', 'name': 'NN'}, {'field': 'Class_1_proba_lb', 'name': 'LB'},
               {'field': 'Comb_proba_1', 'name': 'Ensemble'}]

allstatspath = '/home/aapos/Documents/predictions2020/results'
firefile = '/home/aapos/Documents/predictions2020/firebrigrep2020withid.csv'
dffires = pd.read_csv(firefile)
dffires['Dateonly'] = dffires['Date'].str[0:8]
alldates = len(dffires['Dateonly'].unique())
print("All dates with fires:")
print(dffires['Dateonly'].unique())
print("All dates from fire brigade: %d"%alldates)

start_date = datetime.strptime('20200801', '%Y%m%d')
start_date = datetime.strptime('20200904', '%Y%m%d')

start_date = start_date - timedelta(days=1)
curdate = start_date
FP=0
all=0
allpredicted = 0
alldfjoined = pd.DataFrame()
statcols = ['Corine', 'DEM', 'Slope', 'Aspect', 'Curvature', 'ndvi', 'max_temp', 'min_temp', 'mean_temp',
            'rain_7days', 'res_max', 'dir_max', 'dom_vel', 'dom_dir']
statscols_all = [c+'_sum' for c in statcols] + [c+'_min' for c in statcols] + [c+'_max' for c in statcols]
statspreds = pd.DataFrame()#columns=statscols_all)


while curdate<datetime.now():
    curdate = curdate + timedelta(days=1)
    curdatest = curdate.strftime('%Y%m%d')

    predfile = os.path.join('/home/aapos/Documents/predictions2020/', curdatest, '%s_comb.csv' % curdatest)
    datest = "%s/%s/%s" % (curdatest[6:8], curdatest[4:6], curdatest[2:4])
    if not os.path.exists(predfile):
        if datest in dffires['Dateonly'].unique():
            print('missing date %s with fires'%datest)
        else:
            print('missing date %s' % datest)
        continue
    else:
        print('processing date %s' % datest)

    allpredicted += 1
    dffiresdate = dffires[dffires['Date'].str.contains("%s/%s/%s"%(curdatest[6:8],curdatest[4:6],curdatest[2:4]), regex=False)]
    #dffiresdate = dffires[dffires['Date'].str.contains("04/08", regex=False)]
    if dffiresdate.shape[0]==0:
        print('No fires reported for date %s'%datest)
        continue
    elif dffiresdate[dffiresdate['id'].isnull()].shape[0] == dffiresdate.shape[0]:
        print('No reported fires in mask cells for date %s'%datest)
        continue
    else:
        dffiresdate = dffiresdate[dffiresdate['id'].notnull()]

    dfpreds = pd.read_csv(predfile)

    # min val average for all predictions
    newvals = {}
    for statcol in statcols:
        newvals[statcol+'_sum'] = dfpreds[statcol].sum()
        newvals[statcol+'_cnt'] = dfpreds[statcol].count()
        newvals[statcol+'_min'] = dfpreds[statcol].min()
        newvals[statcol+'_max'] = dfpreds[statcol].max()
        dfnv = pd.DataFrame(newvals, index=[0])
    statspreds = pd.concat([statspreds,dfnv])

    dfjoined = dfpreds.join(dffiresdate.set_index('id'), on='id')#,how='inner')
    dfjoinedinner = dfpreds.join(dffiresdate.set_index('id'), on='id',how='inner')
    all+=len(dfjoinedinner)
    FP+=len(dfjoinedinner[dfjoinedinner["Comb_class_pred"]==1])
    alldfjoined=pd.concat([alldfjoined, dfjoinedinner])

    '''
    for statsrange in statsrangelist:
        daystats = get_stats(dfjoined, groupfields, statsrange['range'])
        statsrange['allstats'] = pd.concat([statsrange['allstats'], daystats])
    '''

print('processing all stats')

colsavg = [c for c in statspreds.columns if c[-4:]=='_sum']
colscnt = [c for c in statspreds.columns if c[-4:]=='_cnt']
colsmin = [c for c in statspreds.columns if c[-4:]=='_min']
colsmax = [c for c in statspreds.columns if c[-4:]=='_max']
statspreds[colsavg].sum().to_frame().T
avg = statspreds[colsavg].sum()/statspreds[colscnt].sum()


alldfjoined = alldfjoined.append(statspreds[colsavg].sum(), ignore_index = True)
alldfjoined = alldfjoined.append(statspreds[colsmin].min(), ignore_index = True)
alldfjoined = alldfjoined.append(statspreds[colsmax].max(), ignore_index = True)
alldfjoined.to_csv(os.path.join(allstatspath, 'allfirefeatures.csv'))
'''
for statsrange in statsrangelist:
    statcols = [c for c in statsrange['allstats'].columns if not 'prediction' in c and not 'act' in c]
    monthstats = statsrange['allstats'][statcols]
    sum_monthstats = monthstats.groupby(monthstats.index).sum()
    calc_percents(sum_monthstats, groupfields)
    store_plot_stats(allstatspath, sum_monthstats, 'Aug_Sep_2020_%d_class' % (len(statsrange['range']) - 1), 'August September 2020')
'''

print("all dates predicted: %d" % allpredicted)
print("all fires match a firehub cell: %d"%all)
print("all fires predicted: %d"%FP)





