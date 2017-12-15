"""
MODULE NAME: MRM_Restore_Sn

DESCRIPTION: tools to restore signals that blong to Sn
             It includes the following codes:

                 - Denoise_FBS


@author: J.Bobin
@version: v0.1  July, 4th
"""

import numpy as np
from copy import deepcopy as dp
from pyRestoreManifold.utils import MakeSignal as mss
import pyRestoreManifold.trans.ManiStarletSn as mrm

#############################################################################
###########   FBS-based algorithm ###########################################
#############################################################################

def Denoise_FBS(X,Yin=None,nmax=100,kmad=3,tol=1e-6,gamma=0.5,J=3,verb=0,L0=False,Fixed=None,wscale=None,WithRef=None):

    """
    Solves  min_{Y in Sn} lambda ||F_Sn(Y)||_1 + 0.5*||X -  Y||_F^2

    """

    if Yin is not None:
        Y = dp(Yin)
    else:
        Y = dp(X)
    Go_On = 1
    it = 0
    dtol = 1.
    L = 1.
    alpha = gamma/L
    Yold = dp(Y)

    f = []

    while Go_On:
        it += 1
        if it > nmax:
            Go_On = 0
        if dtol < tol:
            Go_On = 0

        # Compute the gradient / gradient step

        dg = X-Y

        Y = Y + alpha*dg # The update could also be done on the hypersphere - should not change

        # Projecting onto the hypersphere

        mY = np.maximum(0,1e-32+np.sqrt(np.sum(Y**2,axis=0))) # Keep the modulus constant
        Y = Y/mY

        # Thresholding
        if Fixed is None:
            thf = None
        else:
            thf = alpha*Fixed

        Y = Threshold_Sn(Y,kmad=kmad,J=J,L0=L0,Fixed=thf,wscale=wscale,WithRef=Yin)

        #mY = np.maximum(0,1e-32+np.sqrt(np.sum(Y**2,axis=0))) # Keep the modulus constant
        #Y = Y/mY

        # convergence criterion

        #dtol = np.mean(abs(np.arccos(np.minimum(1.,np.sum(Y*Yold,axis=0))))) # Angular variation
        dtol = np.linalg.norm(Y-Yold)/np.linalg.norm(Yold)
        Yold = dp(Y)
        f.append(dtol)

        if verb:
            print("It. #",it," - dtol = ",dtol)

    return Y,f

### Inpainting

def Inpaint_FBS(X,Mask=None,Yin=None,nmax=100,kmad=3,tol=1e-6,gamma=0.5,J=3,verb=0,L0=False,Fixed=None,wscale=None):
    """
    Solves  min_{Y in Sn} lambda ||F_Sn(Y)||_1 + 0.5*||X -  Y||_F^2

    """

    if Yin is not None:
        Y = dp(Yin)
    else:
        Y = dp(X)
    Go_On = 1
    it = 0
    dtol = 1.
    L = 1.
    alpha = gamma/L
    Yold = dp(Y)

    f = []

    while Go_On:
        it += 1
        if it > nmax:
            Go_On = 0
        if dtol < tol:
            Go_On = 0

        # Compute the gradient / gradient step

        if Mask is None:
            dg = X-Y
        else:
            dg = X-Mask*Y

        Y = Y + alpha*dg # The update could also be done on the hypersphere - should not change

        # Projecting onto the hypersphere

        mY = np.maximum(0,1e-32+np.sqrt(np.sum(Y**2,axis=0))) # Keep the modulus constant
        Y = Y/mY

        # Thresholding
        if Fixed is None:
            thf = None
        else:
            thf = alpha*Fixed

        Y = Threshold_Sn(Y,kmad=kmad,J=J,L0=L0,Fixed=thf,wscale=wscale)

        mY = np.maximum(0,1e-32+np.sqrt(np.sum(Y**2,axis=0))) # Keep the modulus constant
        Y = Y/mY

        # convergence criterion

        #dtol = np.mean(abs(np.arccos(np.minimum(1.,np.sum(Y*Yold,axis=0))))) # Angular variation
        dtol = np.linalg.norm(Y-Yold)/np.linalg.norm(Yold)
        Yold = dp(Y)
        f.append(dtol)

        if verb:
            print("It. #",it," - dtol = ",dtol)

    return Y,f

#############################################################################
###########   GFBS-based algorithm ###########################################
#############################################################################

def Denoise_GFBS(X,Yin=None,nmax=100,kmad=3,tol=1e-6,gamma=0.5,J=3,verb=0,L0=False,Fixed=None,wscale=None):
    """
    Solves  min_{Y in Sn} lambda ||F_Sn(Y)||_1 + 0.5*||X -  Y||_F^2

    """

    if Yin is not None:
        Y = dp(Yin)
    else:
        Y = dp(X)
    Go_On = 1
    it = 0
    dtol = 1.
    L = 1.
    alpha = gamma/L
    Yold = dp(Y)
    w_hs = 0.5
    w_sp = 0.5

    Z_hs = dp(Y)
    Z_sp = dp(Y)

    lamb = np.min([1.5,0.5*(1+2*L/alpha)])

    # Choix des lambda

    f = []

    while Go_On:
        it += 1
        if it > nmax:
            Go_On = 0
        if dtol < tol:
            Go_On = 0

        # Compute the gradient / gradient step

        dg = X-Y
        #Y = Y + alpha*dg # The update could also be done on the hypersphere - should not change

        # Hypersphere constraint

        Zp  = 2*Y - Z_hs + alpha*dg
        Z_hs = Z_hs + lamb*(Zp/np.maximum(0,1e-32+np.sqrt(np.sum(Zp**2,axis=0))) - Y) # Keep the modulus constant

        # Sparsity constraint

        Zsp  = 2*Y - Z_sp + alpha*dg

        mY = np.maximum(0,1e-32+np.sqrt(np.sum(Zsp**2,axis=0))) # Keep the modulus constant
        Zsp = Zsp/mY
        if Fixed is None:
            thf = None
        else:
            thf = lamb/w_sp*Fixed
        Temp = Threshold_Sn(Zsp,kmad=kmad,J=J,L0=L0,Fixed=thf,wscale=wscale)
        Temp = Temp*mY

        Z_sp = Z_sp + lamb*(Temp - Y) # Keep the modulus constant

        # Estimating Y

        Y = w_hs*Z_hs + w_sp*Z_sp

        # convergence criterion

        #dtol = np.mean(abs(np.arccos(np.minimum(1.,np.sum(Y*Yold,axis=0))))) # Angular variation
        dtol = np.linalg.norm(Y-Yold)/np.linalg.norm(Yold)
        Yold = dp(Y)
        f.append(dtol)

        if verb:
            print("It. #",it," - dtol = ",dtol)

    return Y,f

def Inpaint_GFBS(X,Mask=None,Yin=None,nmax=100,kmad=3,tol=1e-6,gamma=0.5,J=3,verb=0,L0=False,Fixed=None,wscale=None):
    """
    Solves  min_{Y in Sn} lambda ||F_Sn(Y)||_1 + 0.5*||X -  Y||_F^2

    """

    if Yin is not None:
        Y = dp(Yin)
    else:
        Y = dp(X)
    Go_On = 1
    it = 0
    dtol = 1.
    L = 1.
    alpha = gamma/L
    Yold = dp(Y)
    w_hs = 0.5
    w_sp = 0.5

    Z_hs = dp(Y)
    Z_sp = dp(Y)

    lamb = np.min([1.5,0.5*(1+2*L/alpha)])

    # Choix des lambda

    f = []

    while Go_On:
        it += 1
        if it > nmax:
            Go_On = 0
        if dtol < tol:
            Go_On = 0

        # Compute the gradient / gradient step

        if Mask is None:
            dg = X-Y
        else:
            dg = X-Mask*Y
        #Y = Y + alpha*dg # The update could also be done on the hypersphere - should not change

        # Hypersphere constraint

        Zp  = 2*Y - Z_hs + alpha*dg
        Z_hs = Z_hs + lamb*(Zp/np.maximum(0,1e-32+np.sqrt(np.sum(Zp**2,axis=0))) - Y) # Keep the modulus constant

        # Sparsity constraint

        Zsp  = 2*Y - Z_sp + alpha*dg

        mY = np.maximum(0,1e-32+np.sqrt(np.sum(Zsp**2,axis=0))) # Keep the modulus constant
        Zsp = Zsp/mY
        if Fixed is None:
            thf = None
        else:
            thf = lamb/w_sp*Fixed
        Temp = Threshold_Sn(Zsp,kmad=kmad,J=J,L0=L0,Fixed=thf,wscale=wscale)
        Temp = Temp*mY

        Z_sp = Z_sp + lamb*(Temp - Y) # Keep the modulus constant

        # Estimating Y

        Y = w_hs*Z_hs + w_sp*Z_sp

        # convergence criterion

        #dtol = np.mean(abs(np.arccos(np.minimum(1.,np.sum(Y*Yold,axis=0))))) # Angular variation
        dtol = np.linalg.norm(Y-Yold)/np.linalg.norm(Yold)
        Yold = dp(Y)
        f.append(dtol)

        if verb:
            print("It. #",it," - dtol = ",dtol)

    return Y,f

#############################################################################
###########   GFBS-based algorithm ###########################################
#############################################################################

def Denoise_GFBS_Rplus_Sn(X,Yin=None,nmax=100,kmad=3,tol=1e-6,gamma=0.5,J=3,verb=0,L0=False,Fixed=None):
    """
    Solves  min_{Y in Sn} lambda ||F_Sn(Y)||_1 + 0.5*||X -  Y||_F^2

    """

    if Yin is not None:
        Y = dp(Yin)
    else:
        Y = dp(X)
    Go_On = 1
    it = 0
    dtol = 1.
    L = 1.
    alpha = gamma/L
    Yold = dp(Y)
    w_hs = 0.5
    w_sp = 0.5

    Z_hs = dp(Y)
    Z_sp = dp(Y)

    lamb = np.min([1.5,0.5*(1+2*L/alpha)])

    # Choix des lambda

    f = []

    while Go_On:
        it += 1
        if it > nmax:
            Go_On = 0
        if dtol < tol:
            Go_On = 0

        # Compute the gradient / gradient step

        dg = X-Y
        #Y = Y + alpha*dg # The update could also be done on the hypersphere - should not change

        # Hypersphere constraint

        Zp  = 2*Y - Z_hs + alpha*dg
        Z_hs = Z_hs + lamb*(Zp/np.maximum(0,1e-32+np.sqrt(np.sum(Zp**2,axis=0))) - Y) # Keep the modulus constant

        # Sparsity constraint

        Zsp  = 2*Y - Z_sp + alpha*dg

        # On the angles

        mY = np.maximum(0,1e-32+np.sqrt(np.sum(Zsp**2,axis=0))) # Keep the modulus constant

        #--- mY should be used as some re-weighting
        Zsp = Zsp/mY
        if Fixed is None:
            thf = None
        else:
            thf = lamb/w_sp*Fixed
        Temp = Threshold_Sn(Zsp,kmad=kmad,J=J,L0=L0,Fixed=thf)

        # On the modulus (-- we should put a positivity constraint on the modulus ...)

        mY = Threshold_Rplus(mY,kmad=kmad,J=J,L0=L0,Fixed=thf)

        Temp = Temp*mY

        Z_sp = Z_sp + lamb*(Temp - Y) # Keep the modulus constant

        # Estimating Y

        #Y = w_hs*Z_hs + w_sp*Z_sp
        Y = Z_sp # positivity on mY could be added

        # convergence criterion

        #dtol = np.mean(abs(np.arccos(np.minimum(1.,np.sum(Y*Yold,axis=0))))) # Angular variation
        dtol = np.linalg.norm(Y-Yold)/np.linalg.norm(Yold)
        Yold = dp(Y)
        f.append(dtol)

        if verb:
            print("It. #",it," - dtol = ",dtol)

    return Y,f

#############################################################################
###########   THRESHOLDING                ###################################
#############################################################################

def Threshold_Sn(X,kmad=3,Wei=None,J=3,L0=False,Fixed=None,wscale=None,WithRef=None): # we should need to reweight
    """
    Thresholding in the Sn_Starlet representation
    """

    if WithRef is not None:
        w = mrm.Forward2D_Sn(X,WithRef,J=J)
    else:
        w = mrm.Forward2D_Sn(X,J=J)
    nW = np.shape(w)
    n_channels = nW[0]-1
    npix = nW[1]
    w_scale = np.ones((J,)) # scaling factor in case the threshold is fixed
    if wscale is not None:
        w_scale = wscale

    for r in range(J):
        c = w[n_channels,:,:,r]
        if Fixed is None:
            th = kmad*cmad(c)
        else:
            th = Fixed*w_scale[r]
        if Wei is not None:
            th = th*Wei[r,:,:]

        if L0:
            c = c*(c - th > 0)
        else:
            c = (c - th)*(c - th > 0)
        w[n_channels,:,:,r] = c

    return mrm.Backward2D_Sn(w)
#
# def Threshold_Rn(X,kmad=3,Wei=None,J=3,L0=False,Fixed=None,wscale=None): # we should need to reweight
#     """
#     Thresholding in the Sn_Starlet representation
#     """
#
#     w = pys.forward(X,J=J)
#     n_channels = (np.shape(X))[0]
#     w_scale = np.ones((J,)) # scaling factor in case the threshold is fixed
#     if wscale is not None:
#         w_scale = wscale
#
#     for r in range(J):
#         for m in range(n_channels):
#             c = w[m,:,:,r]
#             if Fixed is None:
#                 th = kmad*mad(c)
#             else:
#                 th = Fixed*w_scale[r]
#             if Wei is not None:
#                 th = th*Wei[r,:,:]
#             if L0:
#                 c = c*(abs(c) - th > 0)
#             else:
#                 c = (c - th*np.sign(c))*(abs(c) - th > 0)
#             w[m,:,:,r] = c
#
#     return pys.backward(w).squeeze()

#############################################################################
###########   THRESHOLDING                ###################################
#############################################################################
#
# def Threshold_Rplus(X,kmad=3,Wei=None,J=3,L0=False,Fixed=None): # Positivity should be added
#     """
#     Thresholding in the Sn_Starlet representation
#     """
#
#     npix = np.shape(X)[0]
#     w = pys.forward(X.reshape(1,npix,npix),J=J)
#
#     for r in range(J):
#         c = w[0,:,:,r]
#         if Fixed is None:
#             th = kmad*mad(c)
#         else:
#             th = Fixed
#         if Wei is not None:
#             th = th*Wei[r,:,:]
#         if L0:
#             c = c*(abs(c) - th > 0)
#         else:
#             c = (c - th)*(abs(c) - th > 0)
#         w[0,:,:,r] = c
#
#     return pys.backward(w).squeeze()
#

#############################################################################
###########   FBS-based algorithm - Rn-based ################################
#############################################################################

def Denoise_FBS_Rn(X,Yin=None,nmax=100,kmad=3,tol=1e-6,gamma=0.5,J=3,verb=0,L0=False,Fixed=None,wscale=None):
    """
    Solves  min_{Y in Sn} lambda ||F_Sn(Y)||_1 + 0.5*||X -  Y||_F^2

    """

    if Yin is not None:
        Y = dp(Yin)
    else:
        Y = dp(X)
    Go_On = 1
    it = 0
    dtol = 1.
    L = 1.
    alpha = gamma/L
    Yold = dp(Y)

    f = []

    while Go_On:
        it += 1
        if it > nmax:
            Go_On = 0
        if dtol < tol:
            Go_On = 0

        # Compute the gradient / gradient step

        dg = X-Y

        Y = Y + alpha*dg # The update could also be done on the hypersphere - should not change

        # Thresholding
        if Fixed is None:
            thf = None
        else:
            thf = alpha*Fixed

        Y = Threshold_Rn(Y,kmad=kmad,J=J,L0=L0,Fixed=thf,wscale=wscale)

        # convergence criterion

        dtol = np.linalg.norm(Y-Yold)/np.linalg.norm(Yold)
        Yold = dp(Y)
        f.append(dtol)

        if verb:
            print("It. #",it," - dtol = ",dtol)

    return Y,f

### Inpainting

def Inpaint_FBS_Rn(X,Mask=None,Yin=None,nmax=100,kmad=3,tol=1e-6,gamma=0.5,J=3,verb=0,L0=False,Fixed=None):

    """
    Solves  min_{Y in Sn} lambda ||F_Sn(Y)||_1 + 0.5*||X -  Y||_F^2

    """

    if Yin is not None:
        Y = dp(Yin)
    else:
        Y = dp(X)
    Go_On = 1
    it = 0
    dtol = 1.
    L = 1.
    alpha = gamma/L
    Yold = dp(Y)

    f = []

    while Go_On:
        it += 1
        if it > nmax:
            Go_On = 0
        if dtol < tol:
            Go_On = 0

        # Compute the gradient / gradient step

        if Mask is None:
            dg = X-Y
        else:
            dg = Mask*(X-Mask*Y)

        Y = Y + alpha*dg # The update could also be done on the hypersphere - should not change

        # Thresholding
        if Fixed is None:
            thf = None
        else:
            thf = alpha*Fixed

        Y = Threshold_Rn(Y,kmad=kmad,J=J,L0=L0,Fixed=thf)

        # convergence criterion

        #dtol = np.mean(abs(np.arccos(np.minimum(1.,np.sum(Y*Yold,axis=0))))) # Angular variation
        dtol = np.linalg.norm(Y-Yold)/np.linalg.norm(Yold)
        Yold = dp(Y)
        f.append(dtol)

        if verb:
            print("It. #",it," - dtol = ",dtol)

    return Y,f

#############################################################################
###########   UTILS                       ###################################
#############################################################################

def mad(z):
    return np.median(abs(z - np.median(z)))/0.6735

def cmad(z):
    return np.median(abs(z))/0.6735

def SAD(Y0,Y):
    """
    Spectral angular distance in degrees
    """

    H = np.sum(Y*Y0,axis=0)
    H = H/np.maximum(1,abs(H))

    return np.arccos(abs(H))/np.pi*180.
