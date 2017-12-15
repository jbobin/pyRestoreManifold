# -*- coding: utf-8 -*-
"""
MODULE NAME:

DESCRIPTION

@author: J.Bobin
@version:  Thu Jun  1 14:38:13 2017
"""


import numpy as np
import scipy.io as sio
import MRM_Restoration as rs
import MRM_Utils as mss

############################################################
######       Data on Sn
############################################################

def Gen_Sn(npix=32,TMax=15,SNR=20):
    
    m = 5
    Theta = 0.1*np.ones((npix,npix))
    
    Theta[0:np.int(npix/2.),:] = 1.
    Theta[np.int(npix/2.):npix,:] = -1.
    
    Theta = Theta/np.max(Theta)*np.pi/180*TMax
    
    x0 = np.random.randn(m)
    x = np.random.randn(m,npix,npix)
    v = 0.01*np.random.randn(m,npix,npix)
    v = v/np.linalg.norm(v)
    v0 = np.random.randn(m)
    
    for rx in range(npix):
        for ry in range(npix):
            x[:,rx,ry] = np.cos(Theta[rx,ry])*x0 + np.sin(Theta[rx,ry])*v0
            x[:,rx,ry] = x[:,rx,ry]/np.linalg.norm(x[:,rx,ry])
            
    s = np.power(10.,-SNR/20.)
    
    noise = np.random.randn(5,32,32)
    
    xn = x + s*np.linalg.norm(x)/np.linalg.norm(noise)*noise  

    return xn,x,Theta 
    

    
############################################################
######       Data on K+/Sn
############################################################

    
def test_hyperspectral():
    
    data = sio.loadmat('../../Database/Hyperspectral/Moffett128.mat')
    X = data['Y']
    x2 = np.sqrt(np.sum(X**2,axis=0))
    nD = np.shape(X)
    p = np.int(np.sqrt(nD[1]))
    for q in range(nD[0]):
        X[q,:] = X[q,:]/x2
        
        
    c,w,wv = mss.Starlet_Forward(X.reshape(nD[0],p,p),J=1)
    
    w2 = (x2/np.max(x2)).reshape(128,128)
    
    # Filtering
    
    Xout = rs.prox_StarletSn(X.reshape(177,128,128),kmad=3,J=2,W=w2)