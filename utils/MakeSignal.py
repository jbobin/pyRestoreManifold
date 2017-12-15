# -*- coding: utf-8 -*-
"""
MODULE NAME: MakeSnSignal

DESCRIPTION: tools to generate signals on Sn
             It includes the following major codes:

                 - Make_Sn : generates a signal on Sn with Gaussian-like features
                 - Make_Sn_Rp : generates a signal on Sn x R+ with Gaussian-like features
                 - Make_Sn_Rp_RemoteSensing : remote sensing data
                 - Make_Oblique_Rp_RemoteSensing : remote sensing data


@author: J.Bobin
@version: v0.1  July, 3rd
"""

import numpy as np
import ManiUtils as mu
from copy import deepcopy as dp
import scipy.signal as scs

################################################################################################################
################  STARLET TRANSFORM FOR SIGNALS WITH VALUES ON Sn
################################################################################################################

def Make_Sn(n=5,Theta = None,x0 = None,v0 = None,t=128,nc=5,w=5,amp=10.):
    """
    Allows to generate simple datasets on Sn with Gaussian-like features

    INPUT:
        n : dimension do the hypersphere
        t : pixel size
        nc : number of Gaussian features
        w : width of the Gaussian kernel
        amp : maximal angular variation

    OUTPUT:
        x : n x t x t output dataset
        Theta : t x t angle map in degrees
    """

    if Theta is None:
        Theta = np.zeros((t,t))
        for r in range(nc):
            nx = np.int(t*np.random.rand(1))
            ny = np.int(t*np.random.rand(1))
            Theta[nx,ny] += amp/3.*np.random.randn(1)
        Kernel = np.zeros((t,t))
        t2 = np.double(t)/2.
        alpha = np.log(2.)/w**2
        for rx in range(t):
            for ry in range(t):
                Kernel[rx,ry] = np.exp(-alpha*(rx-t2)**2 -alpha*(ry-t2)**2)
        Theta = scs.convolve2d(Theta,Kernel,mode='same',boundary='wrap')
        Theta = amp*Theta/np.max(abs(Theta))
        Theta = Theta*(abs(Theta) > 1e-3)

        x0 = np.random.randn(n)
        x0 = x0/np.linalg.norm(x0)

        v0 = np.random.randn(n)
        v0 = v0 - np.sum(x0*v0)*x0
        v0 = v0/np.linalg.norm(v0)

    else:
        Theta = Theta/np.max(abs(Theta))*amp

    x = np.random.randn(n,t,t)


    for rx in range(t):
        for ry in range(t):
            x[:,rx,ry] = mu.Exp_Sn(x0,v0,Theta[rx,ry]/180.*3.14)

    return x,Theta,x0,v0

def Make_Sn_v2(n=5,t=128,w=5,amp=10.):
    """
    Allows to generate simple datasets on Sn with Gaussian-like features

    INPUT:
        n : dimension do the hypersphere
        t : pixel size
        w : width of the Gaussian kernel
        amp : maximal angular variation

    OUTPUT:
        x : n x t x t output dataset
        Theta : t x t angle map in degrees
    """

    Theta = np.zeros((t,t))
    Theta[:,0:t/2] = 0.
    Theta[:,t/2:t] = 10.

    x0 = np.random.randn(n)
    x0 = x0/np.linalg.norm(x0)
    x = np.random.randn(n,t,t)
    v0 = np.random.randn(n)
    v0 = v0 - np.sum(x0*v0)*x0
    v0 = v0/np.linalg.norm(v0)

    for rx in range(t):
        for ry in range(t):
            x[:,rx,ry] = mu.Exp_Sn(x0,v0,Theta[rx,ry]/180.*3.14)

    return x,Theta

#################################################################################

def Make_Sn_Rp(n=5,t=128,nc=5,w=5,amp=10.):
    """
    Allows to generate simple datasets on SnxRp with Gaussian-like features

    INPUT:
        n : dimension do the hypersphere
        t : pixel size
        nc : number of Gaussian features
        w : width of the Gaussian kernel
        amp : maximal angular variation

    OUTPUT:
        x : n x t x t output dataset
        Theta : t x t angle map in degrees
        Mod : t x t modulus
    """

    Theta = np.zeros((t,t))
    Mod = np.zeros((t,t))
    for r in range(nc):
        nx = np.int(t*np.random.rand(1))
        ny = np.int(t*np.random.rand(1))
        Theta[nx,ny] += amp/3.*np.random.randn(1)
        Mod[nx,ny] += np.random.rand(1)
    Kernel = np.zeros((t,t))
    t2 = np.double(t)/2.
    alpha = np.log(2.)/w**2
    for rx in range(t):
        for ry in range(t):
            Kernel[rx,ry] = np.exp(-alpha*(rx-t2)**2 -alpha*(ry-t2)**2)
    Theta = scs.convolve2d(Theta,Kernel,mode='same',boundary='wrap')
    Theta = amp*Theta/np.max(abs(Theta))
    Mod = scs.convolve2d(Mod,Kernel,mode='same',boundary='wrap')

    x0 = np.random.randn(n)
    x0 = x0/np.linalg.norm(x0)
    x = np.random.randn(n,t,t)
    v0 = np.random.randn(n)
    v0 = v0 - np.sum(x0*v0)*x0
    v0 = v0/np.linalg.norm(v0)

    for rx in range(t):
        for ry in range(t):
            x[:,rx,ry] = mu.Exp_Sn(x0,v0,Theta[rx,ry]/180.*3.14)

    for r in range(n):
        x[r,:,:] = Mod*x[r,:,:]


    return x,Theta,Mod

#################################################################################

def Make_Sn_Rp_RemoteSensing():

    """
    Allows to load a remote sensing dataset

    OUTPUT:
        X : n x t x t output dataset
    """

    import scipy.io as sio

    data = sio.loadmat('/Users/jbobin/Documents/Python/LENA_DEVL/Database/Hyperspectral/Moffett128.mat')
    X = data['Y'].reshape((177,128,128))

    return X

#################################################################################

def Make_SpectralVariability(n=5,t=128):

    """
    Generates data on Oblique + Rp

    INPUT:
        n : number of channels
        t : pixel size

    OUTPUT:
        X : n x t x t output dataset
        Xdust : n x t x t dust datacube
        cib : t x t cib map

    DESCRIPTION:
        the spectrum of cib is known in that case (CMB-like)
        the dust data cube belongs to Sn x Rp

    """

    import sys
    import os
    PYWORK_LOC = os.environ['PYWORK_LOC']
    LOC_PATH = PYWORK_LOC+'/Planck'
    sys.path.insert(1,LOC_PATH+'/Dust/Module/')
    import mbb_utils as mu  ## We might need to specify a path for that
    import astropy.io.fits as pyf

    face = '1'
    dustloc = LOC_PATH+'/Dust/Melis_May17/my353dustFace'+face+'Full_2048.fits'
    tempLoc = LOC_PATH+'/Dust/Melis_May17/tempGNILCface1.fits'
    betaLoc = LOC_PATH+'/Dust/Melis_May17/specGNILCface1.fits'

    Theta = np.zeros((t,t))
    nc = 15
    amp = 1.
    w = 2
    for r in range(nc):
        nx = np.int(t*np.random.rand(1))
        ny = np.int(t*np.random.rand(1))
        Theta[nx,ny] += amp/3.*abs(np.random.randn(1))
    Kernel = np.zeros((t,t))
    t2 = np.double(t)/2.
    alpha = np.log(2.)/w**2
    for rx in range(t):
        for ry in range(t):
            Kernel[rx,ry] = np.exp(-alpha*(rx-t2)**2 -alpha*(ry-t2)**2)
    Theta = scs.convolve2d(Theta,Kernel,mode='same',boundary='wrap')
    Theta = amp*Theta/np.max(abs(Theta))
    Theta = Theta*(abs(Theta) > 1e-3)

    def process_fits(loc):
        ''' Reads in fits files '''
        hdu = pyf.open(loc)
        mjysr = hdu[0].data
        hdu.close()
        return mjysr

    freq  = np.double(np.linspace(30.,3000.,n))  # Between 30 and 3000 GHz

    dust = process_fits(dustloc)
    cib = Theta # It mimics the tSZ
    temp = process_fits(tempLoc)
    beta = process_fits(betaLoc)

    dust = dust[0:t,0:t]
    cib = cib[0:t,0:t]
    temp = temp[0:t,0:t]
    beta = beta[0:t,0:t]

    Xdust = mu.mbb(np.double(beta).reshape(t**2),np.double(temp).reshape(t**2),np.double(freq)).reshape((n,t,t))
    Amp = dust/Xdust[0,:,:]
    X = dp(Xdust)

    for r in range(n):
        Xdust[r,:,:] = Xdust[r,:,:]*Amp
        X[r,:,:] = Xdust[r,:,:] + cib

    return X,Xdust,cib
