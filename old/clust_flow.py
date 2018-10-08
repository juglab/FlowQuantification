from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import numpy as np
from tps_2 import *

def cluster_flow( flowchannel,pos_s,pos_t):
    I1=flowchannel[0]
    I2=flowchannel[1]
     
    x = pos_s[:,0]
    y = pos_s[:,1]
    vx = pos_t[:,0]-pos_s[:,0]
    vy = pos_t[:,1]-pos_s[:,1]
    
    
    k=20
    kmeans=KMeans(n_clusters=k)
    
    lam=10
    
    X=np.stack((x,y,vx*lam,vy*lam),axis=1)
    
    label=kmeans.fit_predict(X)
    
#    plt.figure()
#    plt.imshow(I1)
    
    norm = Normalize()
    
    colormap = cm.inferno
    remove=[]
    for i in range(k):
        
        if np.sum(label==i)<6:
            X=np.stack((x[label==i],y[label==i],vx[label==i]*lam,vy[label==i]*lam),axis=1)
            remove.append(i)
            
            dist=kmeans.transform(X)
            dist[:,remove]=9999999999999999
            new=np.argmin(dist,axis=1)
            label[label==i]=new
    
    point_num=np.zeros(np.shape(label))
    for i in range(k):
        point_num[i]=np.sum(label==i)
        print()
        colors = i
        norm.autoscale(colors)
        color = list(np.random.choice(range(256), size=3)/255)

#        
#        plt.quiver(x[label==i],y[label==i],vx[label==i],vy[label==i],color=color,scale=1,scale_units='x')
        
    
    
    
    
    mesh_x = np.arange(np.shape(flowchannel[0])[0])
    mesh_y = np.arange(np.shape(flowchannel[0])[1])
    mesh_y, mesh_x = np.meshgrid(mesh_x, mesh_y)
    
    
    mesh_xy=np.stack((mesh_x.flatten(),mesh_y.flatten()),axis=1)
     
     
    centroids=kmeans.cluster_centers_ 
    centroids_x=centroids[np.where(point_num>0)[0],0]
    centroids_y=centroids[np.where(point_num>0)[0],1]

    w=np.zeros((np.shape(mesh_x)[0],np.shape(mesh_x)[1],np.shape(centroids_y)[0]))
    
    for index in range(len(mesh_xy)):
        pt = mesh_xy[index,:]   
        distances=np.sqrt((centroids_x-pt[0])**2+(centroids_y-pt[1])**2)
        
        distance_inverse = 1/(distances+0.001)
        normalized_weight_vector= distance_inverse/np.sum( distance_inverse)
        
        w[pt[0],pt[1],:]=normalized_weight_vector

    
#    label_new=[]
#    for i in range(k):
#        if point_num>0:
            
        
    
    u=np.zeros((np.shape(mesh_x)[0],np.shape(mesh_x)[1],np.shape(centroids_y)[0]))
    v=np.zeros((np.shape(mesh_x)[0],np.shape(mesh_x)[1],np.shape(centroids_y)[0]))
    
    q=np.where(point_num>0)[0]
    for ii,i in enumerate(q):
        pos_ss=pos_s[label==i]
        pos_tt=pos_t[label==i]
        
        
        flow=compute_flow_TPS2( flowchannel,pos_ss,pos_tt)[0]
        u[:,:,ii]=flow[:,:,0]
        v[:,:,ii]=flow[:,:,1]
        
        
    uu=np.sum(u*w,axis=2)
    vv=np.sum(v*w,axis=2)   
    
    
    flows=np.stack((uu,vv),axis=2)
    
    return [flows]
        
    
        
        
        
    
        
        
    
    
    
    
    
    


