import numpy as np
import numpy.linalg as nl
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, cdist, squareform

def makeT(cp):
    # cp: [K x 2] control points
    # T: [(K+3) x (K+3)]
    K = cp.shape[0]
    T = np.zeros((K+3, K+3))
    T[:K, 0] = 1
    T[:K, 1:3] = cp
    T[K, 3:] = 1
    T[K+1:, 3:] = cp.T
    R = squareform(pdist(cp, metric='euclidean'))
    R = R * R
    R[R == 0] = 1 # a trick to make R ln(R) 0
    R = R * np.log(R)
    np.fill_diagonal(R, 0)
    T[:K, 3:] = R
    return T

def liftPts(p, cp):
    # p: [N x 2], input points
    # cp: [K x 2], control points
    # pLift: [N x (3+K)], lifted input points
    N, K = p.shape[0], cp.shape[0]
    pLift = np.zeros((N, K+3))
    pLift[:,0] = 1
    pLift[:,1:3] = p
    R = cdist(p, cp, 'euclidean')
    R = R * R
    R[R == 0] = 1
    R = R * np.log(R)
    pLift[:,3:] = R
    return pLift


def compute_flow_TPS2( flowchannel,pos_s,pos_t):
     '''Computes the Thin Plate Splines for the given movie
     '''
     src_img = flowchannel[0]
     des_img = flowchannel[1]
     
     x=pos_s[:,1]
     y=pos_s[:,0]
     
     xs = x.flatten()
     ys = y.flatten()
     cps = np.vstack([xs, ys]).T
     
     
     xt=pos_t[:,1]
     yt=pos_t[:,0]
     
     T = makeT(cps)
     
     xtAug = np.concatenate([xt, np.zeros(3)])
     ytAug = np.concatenate([yt, np.zeros(3)])
     cx = nl.solve(T, xtAug) # [K+3]
     cy = nl.solve(T, ytAug)
     
     
     
     mesh_x = np.arange(src_img.shape[0])
     mesh_y = np.arange(src_img.shape[1])
     x, y= np.meshgrid(mesh_x, mesh_y)
     
     
     xgs, ygs = x.flatten(), y.flatten()
     gps = np.vstack([xgs, ygs]).T
     
     pgLift = liftPts(gps, cps) # [N x (K+3)]
     xgt = np.dot(pgLift, cx.T)
     ygt = np.dot(pgLift, cy.T)
     
#     plt.figure()
#     plt.imshow(src_img)
#     
#     plt.figure()
#     plt.xlim(-2.5, 2.5)
#     plt.ylim(-2.5, 2.5)
#     plt.subplot(1, 2, 1)
#     plt.title('Source')
#     plt.grid()
#     plt.scatter(xs, ys, marker='+', c='r', s=40)
#     plt.scatter(xgs, ygs, marker='.', c='r', s=5)
#     plt.subplot(1, 2, 2)
#     plt.title('Target')
#     plt.grid()
#     plt.scatter(xt, yt, marker='+', c='b', s=40)
#     plt.scatter(xgt, ygt, marker='.', c='b', s=5)
#     plt.show()
     
     u=np.zeros(np.shape(src_img))
     v=np.zeros(np.shape(src_img))
     
     xgs=xgs.astype(np.int)
     ygs=ygs.astype(np.int)
     
     u=xgt-xgs
     v=ygt-ygs
     
#     u[ygs,xgs ]=u
#     v[ygs,xgs]=v
     u=u.reshape(src_img.shape[1],src_img.shape[0])
     v=v.reshape(src_img.shape[1],src_img.shape[0])
     
     
     flows=np.stack((u,v),axis=2)
     return [flows]

