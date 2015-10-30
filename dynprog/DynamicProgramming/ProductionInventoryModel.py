# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 19:33:43 2015

@author: root
"""
import numpy as np
import Helpers as helpers
from scipy.interpolate import RegularGridInterpolator

class ProductionInventoryModel:

    '''
    states and control format
    s,p,q,x = s[0],s[1],a[0],a[1]
    
    --- state ---
    s - stock inventory in [0, 2]
    p - prices in [0.6, 1.4]
    --- controls ---
    q - produce >= 0
    x - store   >= 0
    V(s,p)
    g( s, p, q, x) : (x, h(p,epsilon)) -> (s, p)
    
    '''

    def __init__( self, c1 = 0.5, c2 = 0.1, k1 = 0.1, k2 = 0.1, rho = 0.5, pMean = 1, beta = 0.95 ):
        self.c1 , self.c2, self.k1, self.k2 = c1, c2, k1, k2
        self.rho = rho
        self.pMean = pMean        
        self.beta = beta
        
        def getIdx(grid_min = [0.0,0.5 ], grid_max = [2.0,1.5] , grid_size = [5,20]):        
            return map( lambda x: np.linspace(x[0],x[1],x[2]),zip(grid_min,grid_max,grid_size))        
        
        def cf(q):
            '''
            production cost of q units
            '''
            return c1*q+0.5*c2*(q**2)

        def dcf(q):
            return c1+c2*q            
            
        def kf(x):
            '''
            storage cost of k units to the next iteration
            '''
            return k1*x + 0.5*k2*(x**2)

        def dkf(x):
            return k1 + k2*x

        def utility( s, a ):
            si,p,q,x = s[0],s[1],a[0],a[1]        
            return p*(si+q-x)-cf(q)-kf(x)
    
        def dUtility( s, a):
            p,q,x = s[1],a[0],a[1]        
            return np.array([p-dcf(q),-p-dkf(x)],dtype =float)
    
    
        def g( s, a, e):        
            p,x = s[1],a[1]               
            return np.array([x,
                             pMean+rho*(p-pMean)+e])
                             
        def dG( s, a, e):            
            return np.array([[0,1],
                             [0,0]])
    
        def ua( s ):
            return np.array([1e+6,2],dtype = float)

        def la( s ):
            return np.array([1e-6,0],dtype = float)    
            
        def init( idx ):
            interpolator = RegularGridInterpolator(((0,2),(0.5,1.5)),[[10,15],[12,17]])
            w = np.zeros( (len(idx[0]),len(idx[1])) , dtype = float)        
        
            it = np.nditer(w, flags=['multi_index'], op_flags=['readonly'])
            while not it.finished: 
                pos = map( lambda x, i : idx[i][x] , 
                          it.multi_index, 
                          xrange(len(it.multi_index)) )
                w.itemset(it.multi_index,interpolator(pos))
                it.iternext()
                
            return w      
        
        def weigths( n = 3 , mu = 0 , sigma = 0.2 ):
            e, w = helpers.MathHelpers.GaussNoise(n , mu , sigma )
            return zip(e,w)
        
        self.getIdx = getIdx
        self.cf = cf
        self.kf = kf
        self.utility = utility
        self.dUtility = dUtility
        self.g = g
        self.dG = dG
        self.ua = ua
        self.la = la
        self.init = init
        self.weigths = weigths
       
    def __repr__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)
    
    def __str__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)
       
