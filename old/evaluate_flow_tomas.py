import time
import copy
import numpy as np
from skimage.filters import gaussian
import cv2 
from unknown_met import *
from clust_flow import *

import matplotlib.pyplot as plt

import itertools


from flow_methods import *

from tps_2 import *

#%matplotlib qt5
#%pylab inline
# pylab.rcParams['figure.figsize'] = (15, 10)

from tifffile import imread, imsave

from skimage.draw import circle

import csv

from skimage import io
from scipy.interpolate import griddata
from sklearn.neighbors import KDTree


#from openpiv.tools import imread



def quantify_flows(flows,validation_source_rows,validation_target_rows):
    
    flows=flows[0]
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
    
    return lag, euk, angle, euk_rel, lag_rel







filename = 'imageForTPS.tif'
flow_channel = imread(filename)

flow_channel=np.double(flow_channel)


maxi=np.max(flow_channel)
mini=np.min(flow_channel)

flow_channel= (np.double(flow_channel)-mini)/(maxi-mini)






#flow_channel[1]= (np.double(flow_channel[1])-np.min(flow_channel[0]))/(np.max(flow_channel[0])/8-np.min(flow_channel[0]))*255
#flow_channel= (np.double(flow_channel))/255
img0=flow_channel

img0=img0*6
img0[img0>1]=1


#flow_channel = flow_channel.astype(np.uint8)#for optical flow
#
#flow_channel = np.zeros(A.shape, np.double)

#plot.imshow()

##
#filename = 'MyosinStack.tif'
#flow_channel = imread(filename)[0:5,:,:]
#flow_channel = flow_channel.astype(np.uint8)
#flow_channel = np.double(flow_channel)/255

#flow_channel = np.zeros(A.shape, np.double)
#flow_channel = cv2.normalize(flow_channel, flow_channel, 0.5, 0.0, cv2.NORM_MINMAX)*255





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




### Reading only certain indices (with uncertainty readius = 0) for warping
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
    
    
    
    
k = 3
np.random.seed(0)
r=np.random.choice(len(certain_source_rows), size=len(certain_source_rows), replace=False, p=None)

full_angle = []
full_lag = []
full_euk = []
full_euk_rel = []
full_lag_rel = []

certain_source_rows = certain_source_rows[r,:]
certain_target_rows = certain_target_rows[r,:]
step = int(np.ceil(len(certain_source_rows)/k))
for i in range(k):
    k_fold_validation_indices = np.arange(i*step,i*step+step)
    k_fold_training_indices = np.concatenate((np.arange(i*step),np.arange(i*step+step,len(certain_source_rows))),axis=0)
    
    k_fold_validation_indices=k_fold_validation_indices[k_fold_validation_indices<np.shape(certain_source_rows)[0]]
    k_fold_training_indices=k_fold_training_indices[k_fold_training_indices<np.shape(certain_source_rows)[0]]
    
    training_source_rows = certain_source_rows[k_fold_training_indices,:]
    training_target_rows = certain_target_rows[k_fold_training_indices,:]
    validation_source_rows = certain_source_rows[k_fold_validation_indices,:]
    validation_target_rows = certain_target_rows[k_fold_validation_indices,:]
#    
#    validation_source_rows=training_source_rows
#    validation_target_rows=training_target_rows
##    
    ####Change this optical flow, it's just for testing pipeline
    
#    flows=compute_flow_TPS2( flow_channel,training_source_rows, training_target_rows)
#    flows=optical_flow( flow_channel)
#    flows=mser_nearest( flow_channel,training_source_rows, training_target_rows)
#    flows=random_flow( flow_channel,training_source_rows, training_target_rows)
    flows=mser_TM( flow_channel,training_source_rows, training_target_rows)
#    flows=compute_kNNinterpolated_flow( flow_channel,training_source_rows, training_target_rows,uncertainty)
#    flows=compute_kNNinterpolate_with_optical_flow( flow_channel,training_source_rows, training_target_rows)
#    flows=guided_hornshunk_flow( flow_channel,training_source_rows, training_target_rows)
#    flows=cluster_flow( flow_channel,training_source_rows, training_target_rows)
    

    lag, euk, angle, euk_rel, lag_rel = quantify_flows(flows,validation_source_rows,validation_target_rows)
    full_angle.append(angle.tolist())
    full_lag.append(lag.tolist())
    full_euk.append(euk.tolist())
    full_lag_rel.append(lag_rel.tolist())
    full_euk_rel.append(euk_rel.tolist())
    
#     tmp=validation_target_rows-validation_source_rows
#     u0=tmp[:,0]
#     v0=tmp[:,1]




full_lag = np.asarray([item for sublist in full_lag for item in sublist])
full_angle = np.asarray([item for sublist in full_angle for item in sublist])
full_euk = np.asarray([item for sublist in full_euk for item in sublist])
full_lag_rel = np.asarray([item for sublist in full_lag_rel for item in sublist])
full_euk_rel = np.asarray([item for sublist in full_euk_rel for item in sublist])

    
print("Angle difference:", np.nanmedian(full_angle))
print("Pixel lag:", np.nanmedian(np.abs(full_lag)))
print("Euclidean offset:", np.nanmedian(full_euk))
print("Relative euclidean offset:", np.nanmedian(full_euk_rel))
print("Relative lag:", np.nanmedian(np.abs(full_lag_rel)))


x = np.arange(np.shape(flow_channel[0])[0])
y = np.arange(np.shape(flow_channel[0])[1])
#x, y = np.meshgrid(y, x)#of
y, x = np.meshgrid(x, y)

u=flows[0][:,:,0]
v=flows[0][:,:,1]

subsample=5

x=x[0::subsample,0::subsample]
y=y[0::subsample,0::subsample]
u=u[0::subsample,0::subsample]
v=v[0::subsample,0::subsample]

plt.imshow(img0[0])


plt.quiver(x,y,v,u,color='b',scale=1,scale_units='x')


x=training_target_rows[:,0]
y=training_target_rows[:,1]

tmp=training_target_rows-training_source_rows
u0=tmp[:,0]
v0=tmp[:,1]


plt.quiver(x,y,u0,v0,color='r',scale=1,scale_units='x')


x=validation_target_rows[:,0]
y=validation_target_rows[:,1]


tmp=validation_target_rows-validation_source_rows
u0=tmp[:,0]
v0=tmp[:,1]



plt.quiver(x,y,u0,v0,color='g',scale=1,scale_units='x')


plt.show()

#
#plt.figure()
#np.histogram(full_euk, bins=10)
#
#plt.hist(full_euk, bins=10)  # arguments are passed to np.histogram
#
#plt.show()