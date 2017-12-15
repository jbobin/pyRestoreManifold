# -*- coding: utf-8 -*-
"""
MODULE NAME:

DESCRIPTION

@author: J.Bobin
@version:  Thu Jun  1 14:08:43 2017
"""

import numpy as np
from copy import deepcopy as dp

# Import wrappers
import pyRestoreManifold.wrappers.ManiMR as mmr
from pyRestoreManifold.utils import ManiUtils as mu


def Forward2D_Sn(X,J=2,h=[0.0625,0.25,0.375,0.25,0.0625]):

    """
    Forward starlet transform on Sn
    """

    lh = len(h)
    nX = np.shape(X)
    x = mmr.ManiMR(nX[1],nX[2],nX[0],J,lh,0).forward2d_omp(np.real(X),np.array(h))

    return x

def Backward2D_Sn(X):

    """
    Backward starlet transform on Sn
    """

    lh = 5
    nX = np.shape(X)
    x = mmr.ManiMR(nX[1],nX[2],nX[0]-1,nX[3]-1,lh,0).backward2d_omp(X)

    return x

def Forward1D_Sn(X,J=2,h=[0.0625,0.25,0.375,0.25,0.0625]):

    """
    Forward starlet transform on Sn (TO BE DONE)
    """

    lh = len(h)
    nX = np.shape(X)
    x = mmr.ManiMR(nX[1],nX[2],nX[0],J,lh,0).forward2d_omp(np.real(X),np.array(h))

    return x

def Backward1D_Sn(X):

    """
    Backward starlet transform on Sn (TO BE DONE)
    """

    lh = 5
    nX = np.shape(X)
    x = mmr.ManiMR(nX[1],nX[2],nX[0]-1,nX[3]-1,lh,0).backward2d_omp(X)

    return x


def Forward2D_Rn(X,h = [0.0625,0.25,0.375,0.25,0.0625],J = 1):

    """
    Forward starlet transform on Rn
    """

    nX = np.shape(X)
    Lh = np.size(h)

    W = mmr.Starlet2D(nX[1],nX[2],nX[0],J,Lh).forward_omp(np.real(X),np.array(h))

    return W

def Backward2D_Rn(W,h = [0.0625,0.25,0.375,0.25,0.0625]):

    """
    Backward starlet transform on Rn
    """

    nX = np.shape(W)
    Lh = np.size(h)

    rec = mmr.Starlet2D(nX[1],nX[2],nX[0],nX[3]-1,Lh).backward_omp(np.real(W))

    return rec

def Forward1D_Rn(X,h = [0.0625,0.25,0.375,0.25,0.0625],J = 1):

    """
    Forward starlet transform on Rn
    """

    nX = np.shape(X)
    Lh = np.size(h)

    W = mmr.Starlet2D(nX[1],1,nX[0],J,Lh).forward1d_omp(np.real(X),np.array(h))

    return W

def Backward1d_Rn(W,h = [0.0625,0.25,0.375,0.25,0.0625]):

    """
    Backward starlet transform on Rn
    """

    rec = np.sum(W,axis=2)

    return rec

def adjoint1d(W,h = [0.0625,0.25,0.375,0.25,0.0625]):

    """
    Adjoint starlet transform on Rn
    """

    nX = np.shape(W)
    Lh = np.size(h)

    W = mmr.Starlet2D(nX[1],1,nX[0],nX[2]-1,Lh).adjoint1d(np.real(W),np.array(h))

    return W
