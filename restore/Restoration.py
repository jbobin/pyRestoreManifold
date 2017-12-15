# -*- coding: utf-8 -*-
"""
MODULE NAME: MRM_Restoration

DESCRIPTION: tools for the multiresolution analysis and restoration of of manifold-valued data
             It includes the following major codes:

                 - Threshold_Sn: Thresholding in the Sn-Starlet transform
                 - Denoise_Sn: Denoising based on iterative thresholding
                 - Inpaint_Sn: Inpainting based on iteration thresholding
                 - Deconv_Sn: Deconvolution based on iterative thresholding

                 To be implemented:
                     - The same on SO(3), OB and K+/Sn

@author: J.Bobin
@version: v0.1  Thu Jun  1 14:08:43 2017
"""


import numpy as np
import ManiUtils as mu
from copy import deepcopy as dp
import pyRestoreManifold.utils.ManiUtils as mss
import matplotlib.pyplot as plt

############################################################
######       Code to compute some proximal operators
############################################################

def mad(z,axis=None):

    return np.median(abs(z - np.median(z)))/0.6735

############################################################
######       Code to compute some proximal operators
############################################################

## Positivity

def prox_positivity(X):

    return X*(X > 0)

## Soft-thresholding

def prox_soft(X,W): # With some weight matrix

    Xout = (X - W)*(abs(X - np.median(X)) - W > 0)

    return Xout

## Sn

def prox_Sn(X):

    nX = np.shape(X)
    m = nX[0]

    if len(nX) == 2:
        sX = np.sqrt(np.sum(X**2,axis=0))
        Xout =  X/np.tile(sX,(m,1))

    if len(nX) == 3:
        sX = np.sqrt(np.sum(X**2,axis=0))
        Xout =  X/np.tile(sX,(m,1,1))

    return Xout

## Prox L1 in the n-sphere starlet domain

def prox_StarletSn(X,kmad=3,xref=None,W=None,J=2):

    if xref==None:
        c,w,wv = mss.Starlet_Forward(X,J=J)
    else:
        c,w,wv = mss.Starlet_Forward_ref(X,xref,J=J)

    for r in range(J):
        if W == None:
            w[:,:,r] = prox_soft(w[:,:,r],kmad*mad(w[:,:,r]))
        else:
            w[:,:,r] = w[:,:,r]*W
            w[:,:,r] = prox_soft(w[:,:,r],kmad*mad(w[:,:,r]))
            w[:,:,r] = w[:,:,r]/W

    Xout = mss.Starlet_Backward(c,w,wv)

    return Xout

## Prox L1 in the Euclidean starlet domain

def prox_StarletEuclidean(X,kmad=3,W=None,J=2):

    import Starlet2D as st2

    m = np.shape(X)[0]
    Xout = dp(X)

    for q in range(m):

        c,w = st2.Starlet_Forward2D(X[q,:,:],h=[0.0625,0.25,0.375,0.25,0.0625],J=J,boption=3)

        for r in range(J):

            if W == None:
                w[:,:,r] = prox_soft(w[:,:,r],kmad*mad(w[:,:,r]))
            else:
                w[:,:,r] = prox_soft(w[:,:,r],W[:,:,r,q]) # W should depend on the observation

        Xout[q,:,:] = st2.Starlet_Backward2D(c,w)

    return Xout

############################################################
######       Code to update the weights
############################################################

def UpdateW(X,epsilon=1e-3,kend=0):

    if kend == 0:
        W = epsilon/(epsilon + abs(X)/np.max(abs(X)))
    else:
        thrd = kend*mad(X)
        Xout = X*(abs(X) > thrd)
        W = thrd*epsilon/(epsilon + abs(Xout)/np.max(abs(Xout)))

    return W

############################################################
######       FISTA code for denoising
############################################################


def Denoise_Sn(b,kend = 3,nmax=100,J=3,gamma=1,tol=1e-6,xinit=None,verb=0):

    # Initialize useful parameters

    x = prox_Sn(dp(b))
    if xinit != None:
        x = dp(xinit)

    xold = dp(x)
    tk = 1.
    Go_On = True
    it = 0

    # Main loop

    while Go_On:

        it += 1

        # Compute the gradient of the data fidelity term

        g = x - b

        # Project onto Sn

        xp_half = prox_Sn(x - gamma*g)

        xp = prox_StarletSn(xp_half,kmad=kend,W=None,J=J)

        # Update x

        tkp = 0.5*(1  + np.sqrt(1 + 4*tk**2))

        x = xp + (tk - 1)/tkp*(xp - xold)

        tkp = dp(tk)

        d_x = abs(np.max(np.sum(xp*xold,axis=0))-1)  #-- better adapted to signals that belong to S1
        #d_x = np.linalg.norm(xp - xold)/np.linalg.norm(xold)

        if d_x < tol:
            Go_On = False
        if it > nmax:
            Go_On = False

        if verb:
            print('It. #: ',it,' - Relative variation: ',d_x)

        xold = dp(xp)

    return x

######################################################################
######       FISTA code for denoising with a reference image
######################################################################

def Denoise_Sn_ref(b,xref=None,kend = 3,nmax=100,J=3,gamma=1,tol=1e-6,xinit=None,verb=0,nouter=1):

    # Initialize useful parameters

    x = prox_Sn(dp(b))

    if xinit != None:
        x = dp(xinit)
        xref = dp(xinit)

    xold = dp(x)
    tk = 1.
    Go_On = True
    it = 0

    # Main loop

    for r_outer in range(nouter):

        while Go_On:

            it += 1

            # Compute the gradient of the data fidelity term

            g = x - b

            # Project onto Sn

            xp_half = prox_Sn(x - gamma*g)

            xp = prox_StarletSn(xp_half,kmad=kend,W=None,xref=xref,J=J)  # Acts as some kind of reweighting

            # Update x

            tkp = 0.5*(1  + np.sqrt(1 + 4*tk**2))

            x = xp + (tk - 1)/tkp*(xp - xold)

            tkp = dp(tk)

            d_x = abs(np.max(np.sum(xp*xold,axis=0))-1)  #-- better adapted to signals that belong to S1
            #d_x = np.linalg.norm(xp - xold)/np.linalg.norm(xold)

            if d_x < tol:
                Go_On = False
            if it > nmax:
                Go_On = False

            if verb:
                print('It. #: ',it,' - Relative variation: ',d_x)

            xold = dp(xp)

        if xinit == None:
            print('Updating the ref point')
            xref=dp(x)

    return x

############################################################
######       FISTA code for inpainting
############################################################


def Inpaint_Sn(b,kend = 3,mask=None,nmax=100,J=3,gamma=1,tol=1e-6,xinit=None,xref=None,verb=0):

    #
    #  x belongs to Sn and is sparse in the manifold-based starlet domain
    #

    # Initialize useful parameters

    nb = np.shape(b)
    x = prox_Sn((1.-mask)*np.random.rand(nb[0],nb[1],nb[2]) + mask*b)
    if xinit != None:
        x = dp(xinit)

    xold = dp(x)
    tk = 1.
    Go_On = True
    it = 0

    # Main loop

    while Go_On:

        it += 1

        # Compute the gradient of the data fidelity term

        g = mask*(x - b)

        # Project onto Sn

        xp_half = prox_Sn(x - gamma*g)

        xp = prox_StarletSn(xp_half,kmad=kend,W=None,xref=xref,J=J)

        # Update x

        tkp = 0.5*(1  + np.sqrt(1 + 4*tk**2))

        x = xp + (tk - 1)/tkp*(xp - xold)

        tkp = dp(tk)

        d_x = abs(np.sum(np.sum(xp*xold,axis=0))/(32.**2)-1)  #-- better adapted to signals that belong to S1
        #d_x = np.linalg.norm(xp - xold)/np.linalg.norm(xold)

        if d_x < tol:
            Go_On = False
        if it > nmax:
            Go_On = False

        if verb:
            print('It. #: ',it,' - Relative variation: ',d_x)

        xold = dp(xp)

    return x
