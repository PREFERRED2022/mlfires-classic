import crop_dataset
import geopandas as gpd

#start=time.time()
#print("Start Cropping")
#cropfolder='/mnt/nvme2tb/ffp/datasets/test/2020/attster'
#if not os.path.isdir(cropfolder): os.makedirs(cropfolder)
gdfperif=gpd.read_file(r'/mnt/nvme2tb/ffp/datasets/test/2019/perif/periphereies.shp',encoding='Windows-1253')
gdfperif=crop_dataset.getperif()
for d in range(25,29):
    date='202308'+str(d)
    crop_dataset.cropfile('/mnt/nvme2tb/ffp/datasets/prod/%s/%s_norm.csv'%(date,date),
                      '/mnt/nvme2tb/ffp/datasets/prod/%s/'%date, gdfperif, '_greece',
                      usexyid='id')