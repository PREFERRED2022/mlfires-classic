#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from os import listdir
from os.path import isfile, join
import pandas as pd


# In[2]:


os.chdir('/home/sgirtsou/Documents/June2019/Comb_results')
mypath = '/home/sgirtsou/Documents/June2019/Comb_results'


# In[4]:


onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) and 'comb' in f]
combined_csv = pd.concat([pd.read_csv(f) for f in [onlyfiles[0]]])


# In[5]:


combined_csv.columns


# In[6]:


ranges = [0.0,0.2,0.4,0.6,0.8,1.0]


# In[30]:


combined_cut=pd.cut(combined_csv.Class_1_proba, ranges)


# In[8]:


combined_cut=combined_cut.rename("ranges")


# In[9]:


combined_csv_r=pd.concat([combined_csv,combined_cut],axis=1)


# In[31]:


combined_csv_r


# In[43]:


groups = combined_csv_r.groupby(combined_cut).count()['Class_1_proba']


# In[44]:


groups


# In[52]:


combined_cut_lb=pd.cut(combined_csv.Class_1_proba_lb, ranges)
combined_cut_lb=combined_cut_lb.rename("ranges_lb")
combined_csv_lb_r=pd.concat([combined_csv,combined_cut_lb],axis=1)


# In[53]:


groups_lb = combined_csv_r.groupby(combined_cut_lb).count()['Class_1_proba_lb']
groups_lb


# In[34]:


combined_csv_fire=combined_csv[combined_csv["fire"]==1]
combined_fire_cut=pd.cut(combined_csv_fire.Class_1_proba, ranges)
combined_fire_cut=combined_fire_cut.rename("ranges_fire")


# In[41]:


combined_csv_fire_r=pd.concat([combined_csv_fire,combined_fire_cut],axis=1)


# In[38]:


groups_fire = combined_csv_fire_r.groupby(combined_fire_cut).count()


# In[39]:


allstats=pd.concat([groups["ranges"],groups_fire["ranges_fire"]],axis=1)
type(allstats)


# In[40]:


s1=allstats["ranges"].sum()
s2=allstats["ranges_fire"].sum()
spc1=0
spc2=0
for index, row in allstats.iterrows():
    #print(index,row["ranges"], row["ranges_fire"])
    #print(g["ranges"])
    pc1=row["ranges"]/s1
    spc1+=pc1
    pc2=row["ranges_fire"]/s2
    spc2+=pc2
    print("%s-%s : %3.2f %3.2f"%(index.left,index.right,pc1*100,pc2*100))
#print("0.0-1.0 : %3.2f"%(spc*100))


# In[37]:





# In[ ]:




