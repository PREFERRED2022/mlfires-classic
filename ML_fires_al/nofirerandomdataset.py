'''
Created on 31 Jan 2020

@author: User
'''

import random
import numpy as np
import datetime
from calendar import monthrange
import bisect
import pandas as pd

def get_distrib(counts):
    cd = {c[0]: c[1] for c in counts}
    s = sum(cd.values())
    cdpcs = {c: cd[c] / s for c in cd}
    p = [cdpcs[pc] for pc in cdpcs]
    return p


def get_rand_from_distrib(start, stop, distrib):
    return np.random.choice(np.arange(int(start), int(stop) + 1), p=distrib)


def get_random_date_distrib():

    df_old=pd.read_csv('/mnt/nvme2tb/ffp/datasets/test/fires_new_norm.csv')
    df_allf=df_old[df_old['fire']==1]

    sql = "SELECT to_char(firedate,'MM') firemonth, count(to_char(firedate,'MM'))" + \
          " from %s group by firemonth order by firemonth" % burntable
    df_allf =
    curs.execute(sql)
    month_counts_ls = curs.fetchall()
    mon_distrib = get_distrib(month_counts_ls)

    sql = "SELECT to_char(firedate,'YYYY') fireyear, count(to_char(firedate,'YYYY'))" + \
          " from %s group by fireyear order by fireyear" % burntable
    curs.execute(sql)
    year_counts_ls = curs.fetchall()
    year_distrib = get_distrib(year_counts_ls)

    curs.close()

    return mon_distrib, year_distrib


def get_min_max_year():
    curs = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    sql = "SELECT min(date_part('year', firedate)) miny, max(date_part('year', firedate)) maxy from %s" % burntable
    curs.execute(sql)
    min_max = curs.fetchone()
    min = min_max['miny']
    max = min_max['maxy']

    curs.close()

    return min, max


def get_random_cell_count(not_burned_percent):
    curs = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    sql = "SELECT count(*) from %s" % burntable
    curs.execute(sql)
    burned = curs.fetchone()[0]
    curs.close()
    f = not_burned_percent
    cntrandom = int(f / (1 - f) * burned)
    return cntrandom


def BinarySearch(a, x):
    i = bisect.bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return True
    else:
        return False


def valid_proximity(idc, maskedlist):
    proxyids = [idc - 1, idc - 2228, idc - 2227, idc - 2226, idc + 1, idc + 2228, idc + 2227, idc + 2226]
    return all(BinarySearch(maskedlist, mid) for mid in proxyids)
    # return all(mid in maskedlist for mid in proxyids)


def get_sublist(l, oper, val):
    return [e for e in l if eval("e" + oper + "val")]


def validate_date(idc, burnids, burndict, randate):
    if BinarySearch(burnids, idc):
        # bl = get_sublist(burndict[idc],"<=", randate)
        bl = [e for e in burndict[idc] if e <= randate]
        # al = get_sublist(burndict[idc],">=", randate)
        bdate = max(bl) if bl else None
        # adate = min(al) if al else None
        if (bdate and randate - bdate < datetime.timedelta(days=2 * 365.25)):
            return False
        else:
            return True
    else:
        return True


def select_random_cell(dateslist):
    cntrandom = len(dateslist)
    curs = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    # sql="SELECT mk.id, b.firedate FROM masked mk left join %s b on mk.id=b.id"%burntable
    sql = "SELECT mk.id FROM masked mk"

    randomcells = []
    curs.execute(sql)
    maskedrecs = curs.fetchall()

    maskids = []
    for rec in maskedrecs:
        # maskids.append(rec['id'])
        bisect.insort(maskids, rec['id'])

    curs = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    sql = "SELECT * from %s" % burntable
    curs.execute(sql)
    burnedrecs = curs.fetchall()
    burndict = {}
    burnids = []
    for rec in burnedrecs:
        if BinarySearch(burnids, rec['id']):
            burndict[rec['id']].append(rec['firedate'])
        else:
            burndict[rec['id']] = [rec['firedate']]
            bisect.insort(burnids, rec['id'])

    for i in range(cntrandom):
        rec = random.choice(maskedrecs)
        while not validate_date(rec['id'], burnids, burndict, dateslist[i]) \
                or not valid_proximity(rec['id'], maskids):
            rec = random.choice(maskedrecs)
        randomcells.append([rec['id'], dateslist[i]])

    curs.close()
    return randomcells


def select_random_dates(cntrandom):
    mdist, ydist = get_random_date_distrib()
    miny, maxy = get_min_max_year()
    noburndatelist = []
    for i in range(cntrandom):
        year = get_rand_from_distrib(miny, maxy, ydist)
        month = get_rand_from_distrib(1, 12, mdist)
        daysofmonth = monthrange(year, month)[1]
        day = random.randrange(1, daysofmonth + 1)
        randdate = datetime.date(year=year, day=day, month=month)
        noburndatelist.append(randdate)
    return noburndatelist


def create_table(randomcells):
    curs = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    sql = "truncate table notburned"
    curs.execute(sql)
    conn.commit()

    for cell in randomcells:
        sql = "insert into notburned (id, firedate) VALUES (%d,'%s');" % (cell[0], cell[1])
        curs.execute(sql)

    sql = "update notburned nb set geom=mk.geom from masked mk where mk.id=nb.id"
    curs.execute(sql)

    conn.commit()
    curs.close()


conn = dbconnect()
burntable = 'burned2'


def main():
    not_burned_percent = 0.9

    dateslist = select_random_dates(get_random_cell_count(not_burned_percent))

    randomcells = select_random_cell(dateslist)
    create_table(randomcells)
    conn.close()


if __name__ == "__main__":
    main()



