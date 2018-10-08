import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread, imsave
import csv

from lm_tm import *


def quantify_flows(flows,validation_source_rows,validation_target_rows):
    
    flow_x=flows[:,:,1]
    flow_y=flows[:,:,0]
    u = flow_x[validation_source_rows[:,0],validation_source_rows[:,1]]
    v = flow_y[validation_source_rows[:,0],validation_source_rows[:,1]]
    tmp=validation_target_rows-validation_source_rows
    u0=tmp[:,0]
    v0=tmp[:,1]
    
    len_vec=(u**2+v**2)**0.5
    len_vec0=(u0**2+v0**2)**0.5
    
    euk=((u-u0)**2+(v-v0)**2)**0.5
    
    angle=np.arccos((u*u0+v*v0)/(len_vec*len_vec0+np.finfo(float).eps))/np.pi*180
    
    lag=len_vec-len_vec0
    
    euk_rel= (euk/len_vec0)*100
    lag_rel = (lag/len_vec0)*100
    
    return lag, euk, angle, euk_rel, lag_rel,len_vec0


###read and normalize images
filename = 'imageForTPS.tif'
flow_channel = imread(filename)

flow_channel=np.double(flow_channel)

maxi=np.max(flow_channel)
mini=np.min(flow_channel)

flow_channel= (np.double(flow_channel)-mini)/(maxi-mini)

#better contrast for imshow
img0=flow_channel
img0=img0*6
img0[img0>1]=1






#####read GT data
with open('CSVFiles/Tracks_with_uncertainty_radius.csv', 'r') as f:
    reader = csv.reader(f)
    uncertainty_radius_list_full = list(reader)
uncertainty_radius_list_full = uncertainty_radius_list_full[1:]
uncertainty_radius_list_full=np.array(uncertainty_radius_list_full,np.int)
uncertainty_radius_list_full = [item for sublist in uncertainty_radius_list_full for item in sublist]



uncertainty_radius_list_full = np.array(uncertainty_radius_list_full)
zero_uncertainty_index = np.where(uncertainty_radius_list_full <=1 )[0]
nonzero_uncertainty_index = np.where(uncertainty_radius_list_full >1)[0]

uncertainty=uncertainty_radius_list_full[zero_uncertainty_index]

### Reading only certain indices (with uncertainty readius <=1) for warping
with open('CSVFiles/Choices_Structured.csv') as sd:
    source_reader=csv.reader(sd)
    next(source_reader)
    certain_source_rows=[row for idx, row in enumerate(source_reader) if idx in zero_uncertainty_index]
    certain_source_rows = np.array(certain_source_rows,int)

with open('CSVFiles/Tracks_Structured.csv') as td:
    target_reader=csv.reader(td)
    next(target_reader)
    certain_target_rows=[row for idx, row in enumerate(target_reader) if idx in zero_uncertainty_index]
    certain_target_rows = np.array(certain_target_rows,int)




flows=localmax_TM( flow_channel)


lag, euk, angle, euk_rel, lag_rel,lengths = quantify_flows(flows,certain_source_rows,certain_target_rows)



print("Angle difference:", np.median(angle))
print("Pixel lag:", np.median(np.abs(lag)))
print("Euclidean offset:", np.median(euk))
print("Relative euclidean offset:", np.median(euk_rel))
print("Relative lag:", np.median(np.abs(lag_rel)))
print("Median relative euclidean offset:", np.median(euk)/np.median(lengths)*100)
print("Median relative lag:", np.median(np.abs(lag))/np.median(lengths)*100)


x = np.arange(np.shape(flow_channel[0])[0])
y = np.arange(np.shape(flow_channel[0])[1])
#x, y = np.meshgrid(y, x)#of
y, x = np.meshgrid(x, y)

u=flows[:,:,0]
v=flows[:,:,1]

subsample=5

x=x[0::subsample,0::subsample]
y=y[0::subsample,0::subsample]
u=u[0::subsample,0::subsample]
v=v[0::subsample,0::subsample]


plt.figure()
plt.imshow(img0[0])


plt.quiver(x,y,v,u,color='b',scale=1,scale_units='x')


x=certain_target_rows[:,0]
y=certain_target_rows[:,1]

tmp=certain_target_rows-certain_source_rows
u0=tmp[:,0]
v0=tmp[:,1]


plt.quiver(x,y,u0,v0,color='r',scale=1,scale_units='x')

plt.show()




