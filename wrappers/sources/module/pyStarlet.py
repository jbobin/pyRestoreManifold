import numpy as np
import sparse2d_Sn as sp2

def HypForward(X,J=2,h=[0.0625,0.25,0.375,0.25,0.0625]):

    lh = len(h)
    nX = np.shape(X)
    x = sp2.Starlet2D(nX[1],nX[2],nX[0],J,lh,0).forward2d_omp(np.real(X),np.array(h))

    return x

def HypBackward(X):

    lh = 5
    nX = np.shape(X)
    x = sp2.Starlet2D(nX[1],nX[2],nX[0]-1,nX[3]-1,lh,0).backward2d_omp(X)

    return x

def HypFilter(X,J=2,th=0,L0=0):

    w = HypForward(X,J=J)
    nW = np.shape(w)
    n_channels = nW[0]-1
    npix = nW[1]

    for r in range(J):
        c = w[n_channels,:,:,r]
        if L0:
            c = c*(c - th > 0)
        else:
            c = (c - th)*(c - th > 0)
        w[n_channels,:,:,r] = c

    return HypBackward(w)
