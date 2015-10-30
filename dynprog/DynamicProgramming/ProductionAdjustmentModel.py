# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 11:01:11 2015

@author: root
"""

import numpy as np
import Helpers as helpers
from scipy.interpolate import RegularGridInterpolator

class ProductionInventoryModel:

    '''
    states and control format
    d,l,q = s[0],s[1],a[0]
    
    --- state ---
    d - current demand
    p - lagged prod 
    --- controls ---
    q - current prod
    
    V(d,p) = d P(q) q - c(q) - a(q-l)
    g( s, p, q, e) : (s, p)) -> (e, q)
    
    '''

    def __init__( self, a = 0.5, b = 0.5, c = 0.5, beta = 0.9,
                 n = 3 , mu = 0 , sigma = 0.4):
        self.a , self.b, self.c = a, b, c
        self.beta = beta
        self.lStar = ((1.0-b)/c)**(1.0/b)
        
        ##poprawic z calkowanie 
        def getIdx(grid_min = [1.0,self.lStar-1.0 ], grid_max = [2.0,self.lStar-3.0] , grid_size = [15,10]):        
            return map( lambda x: np.linspace(x[0],x[1],x[2]),zip(grid_min,grid_max,grid_size))        
        
        def cf(q):
            '''
            production cost of q units
            '''
            return c*q
            
        def dcf(q):
            return c
            
        def af(x):
            '''
            storage cost of k units to the next iteration
            '''
            return a*x**2
        
        def daf(x):
            return a*2.0*x

        def Pf(d,q):
            return d*q**(1.0-b)
            
        def dPf(d,q):
            return d*(1.0-b)*q**(-b)
        

        def utility( s, a ):
            d,l,q = s[0],s[1],a[0]
            return Pf(d,q)-cf(q)-af(q-l)
            
        def dUtility( s, a):
            '''
            diff of Utility with respect to a[i]
            (column vector)
            [d U/d x1, d U/d x2]
            '''
            d,l,q = s[0],s[1],a[0]
            return [ dPf(d,q)-dcf(q)-daf(q-l), 0]
    
        def g( s, a, e):        
            q = a[0]              
            return np.array([e,q])
            
        def dG( s, a, e):        
            '''
            diff of G with respect to a[i]
            (matrix)0
            [[d g1/d x1, d g1/d x2],
             [d g2/d x1, d g2/d x2]]
            
            '''            
            return np.array([[0,0],[1,0]])
    
        def ua( s ):
            return np.array([2],dtype = float)

        def la( s ):
            return np.array([1e-6,0],dtype = float)    
            
        def init( idx ):
            interpolator = RegularGridInterpolator(((0,2),(0.5,1.5)),[[10,15],[12,25]])
            w = np.zeros( (len(idx[0]),len(idx[1])) , dtype = float)        
        
            it = np.nditer(w, flags=['multi_index'], op_flags=['readonly'])
            while not it.finished: 
                pos = map( lambda x, i : idx[i][x] , 
                          it.multi_index, 
                          xrange(len(it.multi_index)) )
                w.itemset(it.multi_index,interpolator(pos))
                it.iternext()
                
            return w      
        
        def weigths(  ):
            x, w = helpers.MathHelpers.GaussNoise(n , mu , sigma )
            return zip(x,w)
        
        self.getIdx = getIdx
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
       
