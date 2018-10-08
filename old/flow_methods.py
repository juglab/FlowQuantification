# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 13:37:52 2018

@author: tomas
"""


from matplotlib.pyplot import imshow, pause
import time
import itertools
import cv2
import numpy as np
from scipy.interpolate import griddata
from sklearn.neighbors import KDTree

from scipy.ndimage.filters import laplace
import matplotlib.pyplot as plt

from scipy import signal


def computeDerivatives(im1, im2):
    
    kernel1=np.array([[-1,1],[ -1,1]])
    kernel2=np.array([[-1,-1],[ 1,1]])
    
    fx = signal.convolve2d(im1,0.25*kernel1,'same') + signal.convolve2d(im2, 0.25* kernel1,'same')
    fy = signal.convolve2d(im1, 0.25*kernel2, 'same') + signal.convolve2d(im2, 0.25* kernel2, 'same')
    ft = signal.convolve2d(im1, 0.25*np.ones((2,2)),'same') + signal.convolve2d(im2, -0.25*np.ones((2,2)),'same')
    return fx, fy, ft







def split_flow_components( flows ):
    ''' Receives flow results as computed by 'compute_flow' and returns 
        a tupel of x and y components of this flow results.
    '''
    flow_x = np.moveaxis(np.swapaxes(flows,0,3)[0],-1,0)
    flow_y = np.moveaxis(np.swapaxes(flows,0,3)[1],-1,0)
    return flow_x, flow_y





def kNN_weight_definition(k_value, distance, neighbor_index):
     """ Finds kNN depending on user specified k and defines weights for these neighbors inversly proportional to the 
         distance of neighbors from the queried point.
     """
     weight_list = []
     normalized_weight_vector = []
     sum_of_weights = 0
     distance_squared = distance*distance
     distance_inverse = 1/(distance_squared+0.001)
    
     for i in range (0,k_value):
         weight =  distance_inverse[0][i]
         sum_of_weights =  sum_of_weights+weight
         weight_list.append(weight)
        
     for j in range(0,len(weight_list)):
         normalized_weight = weight_list[j]/sum_of_weights
         normalized_weight_vector.append(normalized_weight)
        
     return normalized_weight_vector

def compute_kNNinterpolated_flow( flowchannel,pos_s,pos_t):
     '''k Nearest neighbor based interpolation
     '''
     k =3
     x = pos_s[:,0]
     y = pos_s[:,1]
     fx = pos_t[:,0]-pos_s[:,0]
     fy = pos_t[:,1]-pos_s[:,1]
     
     
    
     mesh_x = np.arange(np.shape(flowchannel[0])[0])
     mesh_y = np.arange(np.shape(flowchannel[0])[1])
     mesh_y, mesh_x = np.meshgrid(mesh_x, mesh_y)
    
     u=np.zeros(np.shape(mesh_x))
     v=np.zeros(np.shape(mesh_x))
     mesh_xy=np.stack((mesh_x.flatten(),mesh_y.flatten()),axis=1)
     for index in range(len(mesh_xy)):
        pt = mesh_xy[index,:]   
        
        distances_all=np.sqrt((x-pt[0])**2+(y-pt[1])**2)
        
        neighbor_index=np.argsort(distances_all)[0:k]
        distance=distances_all[neighbor_index]
        
        
#        un=uncertainty[neighbor_index]  
        distance_inverse = 1/(distance**2+0.001)#/(un+1)
        normalized_weight_vector= distance_inverse/np.sum( distance_inverse)
         
        
        u_tmp=np.sum(fy[neighbor_index]*normalized_weight_vector)
        v_tmp=np.sum(fx[neighbor_index]*normalized_weight_vector)
        
        if np.min(distance)>100:
            u_tmp=0
            v_tmp=0
        
        u[pt[0],pt[1]]=u_tmp
        v[pt[0],pt[1]]=v_tmp
        
       
     flows=np.stack((u,v),axis=2)
    
     return [flows]




def guided_hornshunk_flow( flowchannel,pos_s,pos_t):

    
    
     
     x_1 = pos_s[:,0]
     y_1 = pos_s[:,1]
     vx = pos_t[:,0]-pos_s[:,0]
     vy = pos_t[:,1]-pos_s[:,1]
     
     I1=flowchannel[0]
     I2=flowchannel[1]
     
     I1=I1*4
     I1[I1>1]=1
     I2=I2*4
     I2[I2>1]=1
     
     
     u_final=np.zeros(np.shape(I1))
     v_final=np.zeros(np.shape(I1))
      
     Vx=np.zeros(np.shape(I1))
     Vy=np.zeros(np.shape(I1))
     for k in range(len(vx)):
        Vx[y_1[k],x_1[k]]=vx[k]
        Vy[y_1[k],x_1[k]]=vy[k]
        
        
        
     I1=I1[300:650,150:460]
     I2=I2[300:650,150:460]
     Vx=Vx[300:650,150:460]
     Vy=Vy[300:650,150:460]
    
    
    
    
     dirak=np.bitwise_or(Vx>0,Vy>0)
    
    
     angle=2*np.pi*np.random.rand(np.shape(dirak)[0],np.shape(dirak)[1]);
    
     u=np.mean(np.sqrt(vx**2+vy**2))*np.sin(angle);
     v=np.mean(np.sqrt(vx**2+vy**2))*np.cos(angle);

#     u=np.zeros(np.shape(dirak))
#     v=np.zeros(np.shape(dirak))
    
     u[dirak] = Vx[dirak];
     v[dirak] = Vy[dirak];
    
     fx, fy, ft = computeDerivatives(I1, I2);
    
     u0=Vx;
     
     v0=Vy;
    
    
    
    
     mu=np.mean(np.sqrt(vx**2+vy**2))
    
     u[0,:]=0
     u[-1,:]=0
     u[:,0]=0
     u[:,-1]=0
    
     v[0,:]=0
     v[-1,:]=0
     v[:,0]=0
     v[:,-1]=0
    
    
    
     iters=10000
    
     for k in range(iters):
         
        vec_len=np.sqrt(u**2+v**2)
         
        print(k)
        delta_u=laplace(u)
        delta_v=laplace(v)
           
        step=0.2
        alpha=0.5
        beta=1
        gama=1
        len_par=-0.00001
        
#        step=0.05
#        alpha=5
#        beta=0
#        gama=3
        
        
#        dE=(gama*dirak*(u-u0)-alpha*delta_u)
        
#        dE=( beta*fx*( fx* u + fy*v + ft )-alpha*delta_u)
        
        dE=gama*dirak*(u-u0)-alpha*delta_u+ beta*fx*( fx* u + fy*v + ft )+len_par*(mu-vec_len)*1/vec_len*u
        
        dE[0,:]=0
        dE[-1,:]=0
        dE[:,0]=0
        dE[:,-1]=0
  
        u= u - step*dE
        
        
#        dE=(gama*dirak*(v-v0)-alpha*delta_v)
        
#        dE=(beta*fx*( fx* v + fy*u + ft )-alpha*delta_u)
        
        dE=gama*dirak*(v-v0)-alpha*delta_v+ beta*fx*( fx* v + fy*u + ft )+len_par*(mu-vec_len)*1/vec_len*v
        
        dE[0,:]=0
        dE[-1,:]=0
        dE[:,0]=0
        dE[:,-1]=0

        v= v - step*dE
        
        
        x = np.arange(np.shape(I1)[0])
        y = np.arange(np.shape(I1)[1])
        y, x = np.meshgrid(x, y)
        
        
        subsample=5
        x=x[0::subsample,0::subsample]
        y=y[0::subsample,0::subsample]
        up=u[0::subsample,0::subsample]
        vp=v[0::subsample,0::subsample]
        
#        
        if k%2000==0 and k!=0:
            
            plt.figure()
            plt.imshow(I1)
            
            plt.quiver(v,u,color='b',scale=1,scale_units='x')
            
            
            plt.quiver(dirak*v0,dirak*u0,color='r',scale=1,scale_units='x')
            
            plt.show()
            plt.pause(1)
        
        
        
     u_final[300:650,150:460]=u
     v_final[300:650,150:460]=v   
            
    

        
     flows=np.stack((v_final.T,u_final.T),axis=2)
    
     return [flows]


def compute_kNNinterpolate_with_optical_flow( flowchannel,pos_s,pos_t):
     '''k Nearest neighbor based interpolation
     '''
     
     
     OF=optical_flow( flowchannel )[0]
     
     k =5
     x = pos_s[:,0]
     y = pos_s[:,1]
     fx = pos_t[:,0]-pos_s[:,0]
     fy = pos_t[:,1]-pos_s[:,1]
    
     mesh_x = np.arange(np.shape(flowchannel[0])[0])
     mesh_y = np.arange(np.shape(flowchannel[0])[1])
     mesh_y, mesh_x = np.meshgrid(mesh_x, mesh_y)
    
     u=np.zeros(np.shape(mesh_x))
     v=np.zeros(np.shape(mesh_x))
     mesh_xy=np.stack((mesh_x.flatten(),mesh_y.flatten()),axis=1)
     for index in range(len(mesh_xy)):
        pt = mesh_xy[index,:]   
        
        distances_all=np.sqrt((x-pt[0])**2+(y-pt[1])**2)
        
        neighbor_index=np.argsort(distances_all)[0:k]
        distance=distances_all[neighbor_index]
        
        
        distance_inverse = 1/(distance**2+0.001)
        normalized_weight_vector= distance_inverse/np.sum( distance_inverse)
         
        x_nearest=x[neighbor_index].astype(np.int)
        y_nearest=y[neighbor_index].astype(np.int)
        
        OF_x=OF[x_nearest,y_nearest,1]
        OF_y=OF[x_nearest,y_nearest,0]
        
        dif_x=-OF_x+fx[neighbor_index]
        dif_y=-OF_y+fy[neighbor_index]
        
        OF_pt_x=OF[pt[0],pt[1],1]
        OF_pt_y=OF[pt[0],pt[1],0]
        
        
        
        u_tmp=np.sum(OF_pt_y+dif_y*normalized_weight_vector)
        v_tmp=np.sum(OF_pt_x+dif_x*normalized_weight_vector)
        
             
#        u_tmp=np.sum(fy[neighbor_index]*normalized_weight_vector)
#        v_tmp=np.sum(fx[neighbor_index]*normalized_weight_vector)

        
        u[pt[0],pt[1]]=u_tmp
        v[pt[0],pt[1]]=v_tmp
        
       
     flows=np.stack((u,v),axis=2)
    
     return [flows]




def compute_flow_TPS( flowchannel,pos_s,pos_t):
     '''Computes the Thin Plate Splines for the given movie
     '''
     src_img = flowchannel[0]
     des_img = flowchannel[1]

     tps = cv2.createThinPlateSplineShapeTransformer()

     sshape = pos_s.astype(np.float32)
     tshape = pos_t.astype(np.float32)
     sshape = sshape.reshape(1,-1,2)
     tshape = tshape.reshape(1,-1,2)


     matches = list()
     for i in range(0, sshape.shape[1],1):
         matches.append(cv2.DMatch(i,i,0))


#     tps.estimateTransformation(tshape,sshape,matches)
     tps.estimateTransformation(sshape,tshape,matches)

     Xpts = np.arange(0, src_img.shape[0])
     Ypts = np.arange(0, src_img.shape[1])
     Points = np.array(list(itertools.product(Xpts, Ypts))).astype( np.float32)
#     mesh_x = np.arange(np.shape(flowchannel[0])[0])
#     mesh_y = np.arange(np.shape(flowchannel[0])[1])
#     mesh_y, mesh_x = np.meshgrid(mesh_x, mesh_y)
#     Points=np.stack((mesh_x.flatten(),mesh_y.flatten()),axis=1)
     Points=Points.reshape(1,-1,2)
#     
#
     ret, tshape_ = tps.applyTransformation (Points)
     Point_skrewd = tshape_[0,:,:]
     Points = Points[0,:,:]
     
     Points=Points.astype(np.int)
     
#     Point_skrewd = tshape_.reshape(src_img.shape[0],src_img.shape[1],2)   
     
     
     ret, tshape_known = tps.applyTransformation(sshape)
     
     
     u=np.zeros(np.shape(src_img))
     v=np.zeros(np.shape(src_img))
     
     u[Points[:,0],Points[:,1]]=Points[:,0]-Point_skrewd[:,0]
     v[Points[:,0],Points[:,1]]=Points[:,1]-Point_skrewd[:,1]
     
     
#     ret, tshape_known = tps.applyTransformation(sshape)


#     Points=Points.reshape(src_img.shape[0],src_img.shape[1],2)    
     
     
#     mesh_x_skr=tps.warpImage(mesh_x)
#     mesh_y_skr=tps.warpImage(mesh_y)

#     flows=Point_skrewd-Points
     
#     flow_x=mesh_x-mesh_x_skr
#     flow_y=mesh_y-mesh_y_skr
#     flows=np.stack((flow_x,flow_y),axis=2)
     flows=np.stack((u,v),axis=2)
     return [flows]
 
    
    

def optical_flow( flowchannel ):
     '''Computes the Farnaback dense flow for the given moview
     '''
     flows = []
     prvs = flowchannel[0]*255
    
   
     for f in range(flowchannel.shape[0]-1):
         print(f)
         nxt = flowchannel[f+1]*255
    
         flow = cv2.calcOpticalFlowFarneback(prev=prvs,
                                             next=nxt,
                                             flow=None,
                                             pyr_scale=0.5, 
                                             levels=2,
                                             winsize=5, #15?
                                             iterations=2,
                                             poly_n=5, 
                                             poly_sigma=1.1, 
                                             flags=0)
         
#         flows.append(flow)
         flows=flow
         prvs = nxt
         print ('.', end="")
     print (' ...done!')

#     flow_x, flow_y = split_flow_components( flows )
     flows=np.stack((flows[:,:,0].T,flows[:,:,1].T),axis=2)
     return [flows]
 
def optical_flow_LK( flowchannel ):
     '''Computes the Farnaback dense flow for the given moview
     '''
     
     src_img = flowchannel[0]
     
     flows = []
     prvs = flowchannel[0]
    
   
     for f in range(flowchannel.shape[0]-1):
         print(f)
         nxt = flowchannel[f+1]
         
#         prvs=prvs.astype(np.float32)
#         nxt=nxt.astype(np.float32)
         
         
         lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
         
         mesh_x = np.arange(src_img.shape[0])
         mesh_y = np.arange(src_img.shape[1])
         y, x= np.meshgrid(mesh_x, mesh_y)
     
     
         xgs, ygs = x.flatten(), y.flatten()
         p0 = np.stack([xgs, ygs],axis=1)
         
         p0=np.expand_dims(p0, axis=1).astype(np.float32)
         
         p1, st, err = cv2.calcOpticalFlowPyrLK(prvs, nxt, p0, None, **lk_params)
         
    
#         flow = cv2.calcOpticalFlowFarneback(prev=prvs,
#                                             next=nxt,
#                                             flow=None,
#                                             pyr_scale=0.5, 
#                                             levels=3,
#                                             winsize=5, #15?
#                                             iterations=2,
#                                             poly_n=5, 
#                                             poly_sigma=1.1, 
#                                             flags=0)
         
         u=p1[:,0,0]-p0[:,0,0]
         v=p1[:,0,1]-p0[:,0,1]
         
    #     u[ygs,xgs ]=u
    #     v[ygs,xgs]=v
         u=u.reshape(src_img.shape[1],src_img.shape[0])
         v=v.reshape(src_img.shape[1],src_img.shape[0])
         
         flows=np.stack((u,v),axis=2)
         prvs = nxt
         print ('.', end="")
     print (' ...done!')
#
#     flow_x, flow_y = split_flow_components( flows )
#     flows=np.stack((flow_x[0,:,:],flow_y[0,:,:]),axis=2)
     return [flows]    
