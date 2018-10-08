import cv2
import numpy as np
import matplotlib.pyplot as plt

import openpiv.tools
import openpiv.process
import openpiv.scaling

from scipy.ndimage import binary_fill_holes as imfill


from skimage.morphology import watershed
from skimage.feature import peak_local_max
from scipy.ndimage import label
from scipy.ndimage.measurements import center_of_mass

from scipy.ndimage.filters import gaussian_filter

from scipy.ndimage import median_filter
from scipy.optimize import linear_sum_assignment


from scipy.spatial import distance_matrix
from flow_methods import *
from scipy.signal import convolve2d

from skimage.feature import peak_local_max


def random_flow( flowchannel,pos_s,pos_t):
    
    
    x_1 = pos_s[:,0]
    y_1 = pos_s[:,1]
    vx = pos_t[:,0]-pos_s[:,0]
    vy = pos_t[:,1]-pos_s[:,1]
    
    angle=2*np.pi*np.random.rand(np.shape(flowchannel[0])[0],np.shape(flowchannel[0])[1]);
        
    u=np.mean(np.sqrt(vx**2+vy**2))*np.sin(angle)
    v=np.mean(np.sqrt(vx**2+vy**2))*np.cos(angle)


       
    flows=np.stack((v.T,u.T),axis=2)
    
    return [flows]




def generate_mser_seg(I1):
    
#    I1=I1*8
#    I1[I1>1]=1

    
#    vis = np.uint8(I1.copy()*255)
#    mser = cv2.MSER_create(_delta = 1,_min_area=30,_max_area = 300,_max_variation = 0.4,_min_diversity = 0.4)
#    regions,q = mser.detectRegions(vis)
##    
##    
##    
##    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
#    hulls=regions
##    
###    cv2.polylines(vis, hulls, 1, (0, 255, 0))
#    regions_img=np.zeros(np.shape(vis))
#    for i in range(len(regions)):
#        regions_img_tmp=np.zeros(np.shape(vis))
#        cv2.fillPoly(regions_img_tmp,pts=[hulls[i]],color=[1,1,1])
##        regions_img_tmp = imfill(regions_img_tmp)
#        regions_img+=regions_img_tmp
###    cv2.imshow('img', vis)
#        
#    local_maxi,num_c = label(peak_local_max(regions_img,min_distance=7, indices=False))
#    regions_img = watershed(-regions_img, markers=local_maxi, mask=regions_img>0,watershed_line=True)
#    
#    
#    c=np.array(center_of_mass(I1,regions_img,np.arange(0,num_c)))
    
    I1[I1<0.02]=0.02
    
    c=peak_local_max(I1, min_distance=3)
    regions_img=[]
    
#    
    return regions_img,c
    



def mser_nearest( flowchannel,pos_s,pos_t):
    
    I1=flowchannel[0]
    I2=flowchannel[1]
    I1=median_filter(I1,3)
    I1=gaussian_filter(I1,0.3)
    
    I2=median_filter(I2,3)
    I2=gaussian_filter(I2,0.3)
    
    
    x = pos_s[:,0]
    y = pos_s[:,1]
    vx = pos_t[:,0]-pos_s[:,0]
    vy = pos_t[:,1]-pos_s[:,1]
    
#    leng=np.sqrt(vx**2+vy**2)
#    
    
    mser1_img,mser1_c=generate_mser_seg(I1)
    mser2_img,mser2_c=generate_mser_seg(I2)
    
    
    
    
    num_p1=np.shape(mser1_c)[0]
    num_p2=np.shape(mser2_c)[0]
    
    
    w=11
    wr=int(np.round((w-1)/2))
    
    bins=10
    
    
    pixel1=np.zeros((num_p1,w**2))
    
    hist1=np.zeros((num_p1,bins))
    for i in range(num_p1):
        posx=int(np.round(mser1_c[i,0]))
        posy=int(np.round(mser1_c[i,1]))
        
        
        pixel=I1[posx-wr:posx+wr+1,posy-wr:posy+wr+1]
        
        pixel1[i,:]=pixel.flatten()
        
        h=np.histogram(pixel.flatten(), bins=bins, range=(0,1), normed=1, weights=None, density=True)[0]
        hist1[i,:]=h
        
        
    hist2=np.zeros((num_p2,bins))    
    pixel2=np.zeros((num_p2,w**2))
    for i in range(num_p2):
        posx=int(np.round(mser2_c[i,0]))
        posy=int(np.round(mser2_c[i,1]))
        
        
        pixel=I2[posx-wr:posx+wr+1,posy-wr:posy+wr+1]
        
        pixel2[i,:]=pixel.flatten()
        
        h=np.histogram(pixel.flatten(), bins=bins, range=(0,1), normed=1, weights=None, density=True)[0]
        hist2[i,:]=h
        
    f1=pixel1
    f2=pixel2
    D_pixel=distance_matrix(f1,f2 )/(w**2)  
        
    
    
    f1=hist1
    f2=hist2


    D_hist=distance_matrix(f1,f2)
    
    
    
    f1=mser1_c
    f2=mser2_c


    D_eukl=distance_matrix(f1,f2 )
    
    D_tresh=9
    D_eukl[D_eukl>D_tresh]=999999999999999
    
    
    
    alp=0
    bet=0
    
    D=alp*D_pixel+D_eukl+ bet*D_hist
    
    
    
    
    row_ind, col_ind=linear_sum_assignment(D)
    dists=D[row_ind, col_ind]
    
    dists_eukl=D[row_ind, col_ind]
    

    row_ind=row_ind[dists_eukl<=D_tresh]
    col_ind=col_ind[dists_eukl<=D_tresh]
    
    pos_ss_add=np.stack((mser1_c[row_ind,1],mser1_c[row_ind,0]),axis=1)
    pos_tt_add=np.stack((mser2_c[col_ind,1],mser2_c[col_ind,0]),axis=1)
    pos_ss=np.concatenate((pos_s,pos_ss_add),axis=0)
    pos_tt=np.concatenate((pos_t,pos_tt_add),axis=0)
#    pos_ss=pos_ss_add
#    pos_tt=pos_tt_add
    
    
    xx = pos_ss[:,0]
    yy = pos_ss[:,1]
    vxx = pos_tt[:,0]-pos_ss[:,0]
    vyy = pos_tt[:,1]-pos_ss[:,1]
    
#    
    plt.figure()
    plt.imshow(mser1_img)
    plt.figure()
    plt.imshow(mser2_img)
    
    plt.figure()
    plt.imshow(I1)
    plt.plot(mser1_c[:,1],mser1_c[:,0],'r.')
    plt.quiver(xx,yy,vxx,vyy,color='b',scale=1,scale_units='x')
    plt.quiver(x,y,vx,vy,color='g',scale=1,scale_units='x')
    
    plt.figure()
    plt.imshow(I2)
    plt.plot(mser2_c[:,1],mser2_c[:,0],'r.')
#    fsdf=sdffd
    flow=compute_kNNinterpolated_flow( flowchannel,pos_ss,pos_tt)
    
    
#    stopka=stop
    return flow
    

def mser_TM( flowchannel,pos_s,pos_t):
    
    
    
    
    
    
    
    I1=flowchannel[0]
    I2=flowchannel[1]
    

    
    
    
    I1=median_filter(I1,5)
    I1=gaussian_filter(I1,1)
    
    I2=median_filter(I2,5)
    I2=gaussian_filter(I2,1)
    
    
    x = pos_s[:,0]
    y = pos_s[:,1]
    vx = pos_t[:,0]-pos_s[:,0]
    vy = pos_t[:,1]-pos_s[:,1]
    
#    leng=np.sqrt(vx**2+vy**2)
#    
    
    mser1_img,mser1_c=generate_mser_seg(I1)
    mser2_img,mser2_c=generate_mser_seg(I2)
    num_p1=np.shape(mser1_c)[0]
#    num_p2=np.shape(mser2_c)[0]
    


    result1x=[]
    result1y=[]
    result2x=[]
    result2y=[]
    
    
    ws=9
    wsr=int(np.round((ws-1)/2))
    wl=19
    wlr=int(np.round((wl-1)/2))
    
    for i in range(num_p1):
        posx=int(np.round(mser1_c[i,0]))
        posy=int(np.round(mser1_c[i,1]))
        
        if posx-wlr<0 or posx+wlr+1>=np.shape(I1)[0] or posy-wlr<0 or posy+wlr+1>=np.shape(I1)[1]:
            continue
        
        
        window_s=I1[posx-wsr:posx+wsr+1,posy-wsr:posy+wsr+1]
        
        window_l=I2[posx-wlr:posx+wlr+1,posy-wlr:posy+wlr+1]
        
        window_s=np.rot90(window_s,2)-np.mean(window_s)
        
        signal=convolve2d(window_l, window_s, mode='same')
        
        signal[:wsr,:]=0
        signal[-wsr:,:]=0
        signal[:,:wsr]=0
        signal[:,-wsr:]=0
        if signal.max()>0:
            ind=np.argwhere(signal.max() == signal)
            x_tmp=ind[0,0]+posx-wlr
            y_tmp=ind[0,1]+posy-wlr
        
            result1x.append([posx])
            result1y.append([posy])
            result2x.append([x_tmp])
            result2y.append([y_tmp])
      
    result1x=np.array(result1x)
    result1y=np.array(result1y)
    result2x=np.array(result2x)
    result2y=np.array(result2y)
        
    pos_ss_add=np.stack((result1y,result1x),axis=1)[:,:,0]
    pos_tt_add=np.stack((result2y,result2x),axis=1)[:,:,0]
#    pos_ss=np.concatenate((pos_s,pos_ss_add),axis=0)
#    pos_tt=np.concatenate((pos_t,pos_tt_add),axis=0)
    pos_ss=pos_ss_add
    pos_tt=pos_tt_add
    
    
    xx = pos_ss[:,0]
    yy = pos_ss[:,1]
    vxx = pos_tt[:,0]-pos_ss[:,0]
    vyy = pos_tt[:,1]-pos_ss[:,1]
    
#    
#    plt.figure()
#    plt.imshow(mser1_img)
#
#    plt.figure()
#    plt.imshow(I1)
#    plt.plot(mser1_c[:,1],mser1_c[:,0],'r.')
#    plt.quiver(xx,yy,vxx,vyy,color='b',scale=1,scale_units='x')
#    plt.quiver(x,y,vx,vy,color='g',scale=1,scale_units='x')
#    
#    plt.figure()
#    plt.imshow(I2)
#    plt.plot(mser2_c[:,1],mser2_c[:,0],'r.')
#    
#    
#    bla=bamdfsdfji
    
#    flow=compute_kNNinterpolated_flow( flowchannel,pos_ss,pos_tt)
    flow=guided_hornshunk_flow( flowchannel,pos_ss,pos_tt)
#    stopka=stop
    return flow
        
    
    