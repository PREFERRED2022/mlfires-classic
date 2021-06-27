from environment import Environment
from notification import notification
from datetime import datetime, timedelta
import os
import argparse
import traceback
import zipfile

def runpredictions():
    #/ home / sgirtsou / automate_predictions.sh 1
    pass

def getmapfile(env, argdate):
    if argdate is None:
        mapdate = datetime.today() + timedelta(days=1)
    else:
        mapdate = argdate
    datest = mapdate.strftime('%Y%m%d')
    dateform = mapdate.strftime('%d-%m-%Y')
    mapfile = '%s%s'%(datest,env.mapfilesuffix)
    mapsubpath = os.path.join(datest, mapfile)
    mappath = os.path.join(env.mapspath, mapsubpath)
    return mappath, dateform

def sendmap(env, mapfile, datest):
    mess = 'Xάρτης εκίμησης κινδύνου %s'%datest
    mess += '\n\nΑυτό είναι ένα αυτοματοποιημένο μήνυμα. '
    mess += 'Για πληροφορίες απευθυνθείτε: \n'+\
            'Charalampos Kontoes : Kontoes@noa.gr\n'+\
            'Stella Girtsou : sgirtsou@noa.gr\n'+\
            'Alex Apostolakis: alex.apostolakis@noa.gr'
    mess += '\n\nΗ εκτίμηση κινδύνου που εμφανίζεται στον χάρτη είναι υπό αξιολόγηση στα πλαίσια ερευνητικού έργου.'
    #print(mess)
    notification(env).send_notification('Risk Map', 'Send', mess, titlesuf=datest, attachments=[mapfile])

def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files + dirs:
            ziph.write(os.path.join(root, file),\
                       os.path.relpath(os.path.join(root, file), os.path.join(path, os.path.pardir)),\
                       compress_type=zipfile.ZIP_DEFLATED)

def zipfiles(filelist, ziph):
    # ziph is zipfile handle
    for f in filelist:
        ziph.write(f, os.path.relpath(f, os.path.join(os.path.dirname(f), os.path.pardir)),\
                   compress_type=zipfile.ZIP_DEFLATED)

def zipmap(mapfile):
    zipmapfile = mapfile+'.zip'
    zipf = zipfile.ZipFile(zipmapfile, mode='w', allowZip64=True)
    zipfiles([mapfile], zipf)
    zipf.close()
    return zipmapfile

def sendfailurenotice(env, mess):
    print(mess)
    notification(env).send_notification('Risk Map', 'Fail to send', mess)

def sendriskmap(mapdate, compress=True):
    try:
        env = Environment()
        dateform = None
        mapfile, dateform = getmapfile(env, mapdate)
        print('Searching for map file path : %s'%mapfile)
        #runpredictions()
        if os.path.exists(mapfile):
            if compress:
                print('Compressing map : %s' % mapfile)
                mapfile = zipmap(mapfile)
            print('Sending map : %s' % mapfile)
            sendmap(env, mapfile, dateform)
        else:
            mess = 'Map not found : %s' % mapfile
            sendfailurenotice(env, mess)
    except:
        mess = 'Error sending map for %s: \n'%dateform+traceback.format_exc()
        sendfailurenotice(env, mess)

def getargs():
    parser = argparse.ArgumentParser(description='Fire Risk Map sending')
    parser.add_argument('-d','--map-date', type=lambda s: datetime.strptime(s, '%d-%m-%Y'),
                        help='The date of the Fire risk Map in format : DD-MM-YYYY. Default is tomorrow '
                                         'date')
    parser.add_argument('-nc, --no-compress', action='store_false',
                        help='Compress map file')
    args = parser.parse_args()
    return vars(args)

def main():
    args = getargs()
    sendriskmap(args['map_date'], args['nc, __no_compress'])

if __name__ == "__main__":
    main()


