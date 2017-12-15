# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 07:59:05 2017

@author: jbobin
"""

import numpy as np
import scipy.linalg as lng
from copy import deepcopy as dp

################################################
# Defines exp and log maps for useful manifolds
################################################

### Sn

def Exp_Sn(xref,v,theta):

    """
     Exp-map of the n-sphere
    """

    m = len(xref)

    # Projecting onto the xref,v plane

    F = np.zeros((m,2))

    F[:,0] = xref
    F[:,1] = v
    F = np.dot(F,RotMatrix(-theta))

    return F[:,0]

def Log_Sn(xref,x):

    """
     Log-map of the 1-sphere
    """

    nX = np.shape(x)

    m = nX[0]
    t = nX[1]

    G = np.zeros((t,))
    Gv = np.zeros((m,t))

    for r in range(t):

        # Correct for permuations

        Xout = dp(x[:,r])

        a = np.sum(Xout*xref)/np.sqrt(np.sum(xref**2)*np.sum(Xout**2)) # Should have unit L2 norm

        if a > 1:
            a = 1
        if a < -1:
            a = -1

        G[r] = np.arccos(a)  # Computing the angles

        v = Xout - a*xref
        Gv[:,r] = v / (1e-24 + np.linalg.norm(v))   # Unit vector in the tangent subspace

    return G,Gv

def Log_Sn_vec(xref,x):

    """
     Log-map of the n-sphere for 1D inputs
    """

    Xout = dp(x)
    a = np.sum(Xout*xref)/np.sqrt(np.sum(xref**2)*np.sum(Xout**2)) # Should have unit L2 norm

    if a > 1:
        a = 1
    if a < -1:
        a = -1

    G= np.arccos(a)  # Computing the angles

    v = Xout - a*xref
    Gv = v / (1e-24 + np.linalg.norm(v))   # Unit vector in the tangent subspace

    return G,Gv

def Log_Sn_mat(xref,x):

    """
     Log-map of the n-sphere for 2D inputs
    """

    nX = np.shape(x)

    m = nX[0]
    t = nX[1]

    G = np.zeros((t,t))
    Gv = np.zeros((m,t,t))

    for rx in range(t):
        for ry in range(t):

            # Correct for permuations

            Xout = dp(x[:,rx,ry])

            a = np.sum(Xout*xref[:,rx,ry])/np.sqrt(np.sum(xref[:,rx,ry]**2)*np.sum(Xout**2)) # Should have unit L2 norm

            if a > 1:
                a = 1
            if a < -1:
                a = -1

            G[rx,ry] = np.arccos(a)  # Computing the angles

            v = Xout - a*xref[:,rx,ry]
            Gv[:,rx,ry] = v / (1e-24 + np.linalg.norm(v))   # Unit vector in the tangent subspace

    return G,Gv

def RotMatrix(theta):

    """
     Rotation matrix
    """

    M = np.zeros((2,2))
    M[0,0] = np.cos(theta)
    M[1,1] = np.cos(theta)
    M[0,1] = np.sin(theta)
    M[1,0] = -np.sin(theta)

    return M


def Exp_OB_S1(xref,v,theta):

    """
     Exp-map of the 1-sphere
    """

    nX = np.shape(xref)
    m = nX[0]
    n = nX[1]

    # Projecting onto the xref,v plane

    F = np.zeros((m,2))
    Xout = dp(xref)

    for q in range(n):

        F[:,0] = xref[:,q]
        F[:,1] = v[:,q]
        F = np.dot(F,RotMatrix(theta[q]))
        Xout[:,q] = F[:,0]

    return Xout

def Log_OB_S1_vec(xref,x):

    """
     Log-map of the 1-sphere for 1D inputs
    """

    nX = np.shape(x)

    m = nX[0]
    n = nX[1]
    t = nX[2]

    G = np.zeros((n,t))
    Gv = np.zeros((m,n,t))

    for r in range(t):

        # Correct for permuations

        Xout = dp(x[:,:,r])

        for q in range(n):

            a = np.sum(Xout[:,q]*xref[:,q,r])/np.sqrt(np.sum(xref[:,q,r]**2)*np.sum(Xout[:,q]**2)) # Should have unit L2 norm
            G[q,r] = np.arccos(a)  # Computing the angles
            if a > 1:
                a = 1
            if a < -1:
                a = -1

            v = Xout[:,q] - a*xref[:,q,r]
            Gv[:,q,r] = v / (1e-24 + np.linalg.norm(v))   # Unit vector in the tangent subspace

    return G,Gv

def Log_OB_S1_mat(xref,x):

    """
     Log-map of the 1-sphere for 2D inputs
    """

    nX = np.shape(x)

    m = nX[0]
    n = nX[1]
    t = nX[2]

    G = np.zeros((n,t,t))
    Gv = np.zeros((m,n,t,t))

    for rx in range(t):
        for ry in range(t):

            # Correct for permuations

            Xout = dp(x[:,:,rx,ry])

            for q in range(n):

                a = np.sum(Xout[:,q]*xref[:,q,rx,ry])/np.sqrt(1e-12 + np.sum(xref[:,q,rx,ry]**2)*np.sum(Xout[:,q]**2)) # Should have unit L2 norm
                if a > 1:
                    a = 1
                if a < -1:
                    a = -1
                G[q,rx,ry] = np.arccos(a)  # Computing the angles
                v = Xout[:,q] - a*xref[:,q,rx,ry]
                Gv[:,q,rx,ry] = v / (1e-24 + np.linalg.norm(v))   # Unit vector in the tangent subspace

    return G,Gv


def Log_OB_S1(xref,x):

    """
     Log-map of the 1-sphere
    """

    nX = np.shape(x)

    m = nX[0]
    n = nX[1]
    t = nX[2]

    G = np.zeros((n,t))
    Gv = np.zeros((m,n,t))

    for r in range(t):

        # Correct for permuations

        #Xout,PiA= CorrectPerm(xref,x[:,:,r])
        Xout = dp(x[:,:,r])

        for q in range(n):

            a = np.sum(Xout[:,q]*xref[:,q])/np.sqrt(np.sum(xref[:,q]**2)*np.sum(Xout[:,q]**2)) # Should have unit L2 norm
            if a > 1:
                a = 1
            if a < -1:
                a = -1
            G[q,r] = np.arccos(a)  # Computing the angles
            v = Xout[:,q] - a*xref[:,q]
            Gv[:,q,r] = v / (1e-12 + np.linalg.norm(v))   # Unit vector in the tangent subspace

    return G,Gv

def Exp_OB(xref,theta):

    """
     Exp-map of the oblique manifold
    """

    F = xref + theta
    nF = np.shape(F)

    for r in range(nF[1]):
        F[:,r] = F[:,r]/np.linalg.norm(F[:,r])

    return F


def Log_OB(xref,x):

    """
     Log-map of the oblique manifold
    """

    nX = np.shape(x)

    m = nX[0]
    n = nX[1]
    t = nX[2]

    G = np.zeros((m,n,t))

    for r in range(t):

        # Correct for permuations

        Xout,PiA= CorrectPerm(xref,x[:,:,r])

        G[:,:,r] = Xout - np.dot(xref,np.dot(PiA,Xout))

    return G

def Mat_CorrectPerm(X0,X):

    """
     Code for correcting permutations for matrix-valued data
    """

    Xout = dp(X)

    nX = np.shape(X)

    for rx in range(nX[2]):
        for ry in range(nX[3]):
            Xt = X[:,:,rx,ry]
            xx,p=CorrectPerm(X0,Xt)
            Xout[:,:,rx,ry]=xx

    return Xout

#
#
#

def CorrectPerm(X0,X):

    """
     Code for correcting permutations
    """

    PiA = np.dot(lng.inv(np.dot(X0.T,X0)),X0.T)
    Diff = np.dot(PiA,X)

    z = np.shape(X)

    for ns in range(0,z[1]):
        Diff[ns,:] = abs(Diff[ns,:])/max(abs(Diff[ns,:]))

    Xout = np.ones(z)

    for ns in range(0,z[1]):
        Xout[:,np.nanargmax(Diff[ns,:])] = X[:,ns]

    return Xout,PiA
