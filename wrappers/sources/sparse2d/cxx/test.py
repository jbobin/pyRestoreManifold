# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 20:25:35 2017

@author: jbobin
"""

import numpy as np

Lh = 5

m2 = np.floor(Lh/2)
N = 10
scale =0

for i in range(N):
    val = 0
    
    for j in range(Lh):
        
        print(i,j)
        
        Lindix = i + np.power(2.,scale)*(j - m2)
        
        if (Lindix < 0):
            
            Lindix = -Lindix
        
        if (Lindix > N-1):
            Lindix = 2*(N - 1) - Lindix
            
        print(Lindix)
        
        