import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage.filters import laplace
from scipy.ndimage.filters import gaussian_filter
def computeDerivatives(im1, im2):
    
    kernel1=np.array([[-1,1],[ -1,1]])
    kernel2=np.array([[-1,-1],[ 1,1]])
    
    fx = signal.convolve2d(im1,0.25*kernel1,'same') + signal.convolve2d(im2, 0.25* kernel1,'same')
    fy = signal.convolve2d(im1, 0.25*kernel2, 'same') + signal.convolve2d(im2, 0.25* kernel2, 'same')
    ft = signal.convolve2d(im1, 0.25*np.ones((2,2)),'same') + signal.convolve2d(im2, -0.25*np.ones((2,2)),'same')
    return fx, fy, ft





def kNN_interpolated_flow( flowchannel,pos_s,pos_t,k=5):
    '''k Nearest neighbor based interpolation
     '''
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

    return flows



def guided_hornshunk_flow( flowchannel,pos_s,pos_t):

    
     flow_init=kNN_interpolated_flow( flowchannel,pos_s,pos_t,k=5)
     flow_init=flow_init[:,:,(1,0)]
     
#     flow_1nn=kNN_interpolated_flow( flowchannel,pos_s,pos_t,k=1)
     
     x_1 = pos_s[:,0]
     y_1 = pos_s[:,1]
     vx = pos_t[:,0]-pos_s[:,0]
     vy = pos_t[:,1]-pos_s[:,1]
     
     I1=flowchannel[0]
     I2=flowchannel[1]
     
     I1=I1*6
     I1[I1>1]=1
     I2=I2*6
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
    
    
     sigma=2
    

     x, y = np.meshgrid(np.arange(-2*sigma,2*sigma), np.arange(-2*sigma,2*sigma))
     d = np.sqrt(x*x+y*y)
     sigma, mu = sigma, 0.0
     g = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
    
    
     dirak=np.bitwise_or(Vx>0,Vy>0)
     dirak=signal.convolve2d(dirak,g,'same')
    
    
#     angle=2*np.pi*np.random.rand(np.shape(dirak)[0],np.shape(dirak)[1]);
#    
#     u=np.mean(np.sqrt(vx**2+vy**2))*np.sin(angle);
#     v=np.mean(np.sqrt(vx**2+vy**2))*np.cos(angle);
     
     

     u=flow_init[:,:,0].T
     v=flow_init[:,:,1].T
     
     u=u[300:650,150:460]
     v=v[300:650,150:460]
    

#     u=np.zeros(np.shape(dirak))
#     v=np.zeros(np.shape(dirak))
    
#     u[dirak] = Vx[dirak]
#     v[dirak] = Vy[dirak]
#    
     
     fx, fy, ft = computeDerivatives(I1, I2)
    
    
    
     u0=u;
     
     v0=v;
    
    
    
    
     mu=np.mean(np.sqrt(vx**2+vy**2))
    
     u[0,:]=0
     u[-1,:]=0
     u[:,0]=0
     u[:,-1]=0
    
     v[0,:]=0
     v[-1,:]=0
     v[:,0]=0
     v[:,-1]=0
    
    
    
     iters=5000
    
     for k in range(iters):
         
        vec_len=np.sqrt(u**2+v**2)
         
        print(k)
        delta_u=laplace(u)
        delta_v=laplace(v)
           
        step=0.1
        alpha=0.2
        beta=0
        gama=1
        len_par=-0.0001
        
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
        if k%1000==0:
            
            plt.figure()
            plt.imshow(I1)
            
            plt.quiver(v,u,color='b',scale=1,scale_units='x')
            
            
            plt.quiver(dirak*v0,dirak*u0,color='r',scale=1,scale_units='x')
            
            plt.show()
            plt.pause(1)
        
        
        
     u_final[300:650,150:460]=u
     v_final[300:650,150:460]=v   
            
    

        
     flows=np.stack((v_final.T,u_final.T),axis=2)
    
     return flows



    
    


