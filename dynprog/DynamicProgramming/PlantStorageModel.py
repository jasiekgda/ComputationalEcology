# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 12:51:51 2015

@author: root
"""

import numpy as np
import Helpers as helpers
from scipy.interpolate import RegularGridInterpolator

         
class PlantStorageModelV1:

    '''
    states and control format
    st,w,a0,a1 = s[0],s[1],a[0],a[1]
    
    --- state ---
    st - storage in [0, 2]
    w - biomas in [0, 2]
    --- controls ---
    a0 - amount of energy taken from storage in [0, 1)]
    a1 - proportion of energy allocated in growth [0,1]
    
    V(s,w)
    utility_t = a0*(1-a1)
    
    g( s, w, a0, a1,e) : [s = (1-a0)(s+P(w)), w = e( w +a0*a1)]
    
    P(w): produkcja energii
    
    '''

    def __init__( self, P , beta = 0.95 ):        
        self.beta = beta
        
        def getIdx(grid_min = [0.0,0.0 ], grid_max = [2.0,2.0] , grid_size = [11,11]):        
            return map( lambda x: np.linspace(x[0],x[1],x[2]),zip(grid_min,grid_max,grid_size))
            
        
        def utility( s, a ):
            a0,a1 = a[0],a[1]
            st,w = s[0],s[1]
            stPw=(st+P(w))
            return a0*(1.0-a1)*stPw
    
        def dUtility( s, a):
            a0,a1 = a[0],a[1]
            st,w = s[0],s[1]
            stPw=(st+P(w))
            
            return np.array([(1.0-a1)*stPw,-a0*stPw],dtype =float)
    
    
        def g( s, a, e):        
            st,w,a0,a1 = s[0],s[1],a[0],a[1]
            stPw=(st+P(w))
            return np.array([(1-a0)*stPw,
                             e*(w+a0*a1*stPw)])
                             
        def dG( s, a, e):            
            a0,a1 = a[0],a[1]
            st,w = s[0],s[1]
            stPw=(st+P(w))
            return np.array([[-stPw,0],
                             [e*a1*stPw,e*a0*stPw]])
    
        def ua( s ):                       
            return np.array([1,1],dtype = float)

        def la( s ):
            return np.array([0,0],dtype = float)    
            
        def init( idx ):
            interpolator = RegularGridInterpolator(((0,8),(0,8)),[[0,8],[12,16]])
            w = np.zeros( (len(idx[0]),len(idx[1])) , dtype = float)        
        
            it = np.nditer(w, flags=['multi_index'], op_flags=['readonly'])
            while not it.finished: 
                pos = map( lambda x, i : idx[i][x] , 
                          it.multi_index, 
                          xrange(len(it.multi_index)) )
                w.itemset(it.multi_index,interpolator(pos))
                it.iternext()
                
            return w      
        
        #def weigths( n = 3 , mu = 0 , sigma = 0.2 ):
        #    e, w = helpers.MathHelpers.GaussNoise(n , mu , sigma )
        #    return zip(e,w)
        
        self.getIdx = getIdx
        self.utility = utility
        self.dUtility = dUtility
        self.P = P
        self.g = g
        self.dG = dG
        self.ua = ua
        self.la = la
        self.init = init        
       
    def __repr__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)
    
    def __str__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)
        
    def ver(self):
        return 8
        
class PlantStorageModelV2:

    '''
    states and control format
    st,w,a0,a1 = s[0],s[1],a[0],a[1]
    
    --- state ---
    st - storage in [0, 2]
    w - biomas in [0, 2]
    --- controls ---
    a0 - amount of energy taken from storage in [0, 1)]
    a1 - proportion of energy allocated in growth [0,1]
    
    V(s,w)
    utility_t = a1 (s+P(w))
    
    g( s, w, a0, a1,e) : [s = (1-a0-a1)(s+P(w)), w = e( w +a0(s+P(w)) )]
    
    P(w): produkcja energii
    
    '''

    def __init__( self, P , beta = 0.95 ):        
        self.beta = beta
        
        def getIdx(grid_min = [0.0,0.0 ], grid_max = [2.0,2.0] , grid_size = [11,11]):        
            return map( lambda x: np.linspace(x[0],x[1],x[2]),zip(grid_min,grid_max,grid_size))
            
        
        def utility( s, a ):
            a1 = a[1]
            st,w = s[0],s[1]
            stPw=(st+P(w))
            return a1*stPw
    
        def dUtility( s, a):            
            st,w = s[0],s[1]
            stPw=(st+P(w))
            
            return np.array([0,stPw],dtype =float)
    
    
        def g( s, a, e):        
            st,w,a0,a1 = s[0],s[1],a[0],a[1]
            stPw=(st+P(w))
            return np.array([(1-a0-a1)*stPw,
                             e*(w+a0*stPw)])
                             
        def dG( s, a, e):    
            st,w = s[0],s[1]
            stPw=(st+P(w))
            return np.array([[-stPw,-stPw],
                             [e*stPw,0]])
    
        def ieqcons( s, a ):
            return 1.0 - a[0] - a[1]

        def dIeqcons ( s, a ):
            return np.array([[-1,-1]])


        def ua( s ):                       
            return np.array([1,1],dtype = float)

        def la( s ):
            return np.array([0,0],dtype = float)    
            
        def init( idx ):
            interpolator = RegularGridInterpolator(((0,8),(0,8)),[[0,8],[12,16]])
            w = np.zeros( (len(idx[0]),len(idx[1])) , dtype = float)        
        
            it = np.nditer(w, flags=['multi_index'], op_flags=['readonly'])
            while not it.finished: 
                pos = map( lambda x, i : idx[i][x] , 
                          it.multi_index, 
                          xrange(len(it.multi_index)) )
                w.itemset(it.multi_index,interpolator(pos))
                it.iternext()
                
            return w      
            
        
        #def weigths( n = 3 , mu = 0 , sigma = 0.2 ):
        #    e, w = helpers.MathHelpers.GaussNoise(n , mu , sigma )
        #    return zip(e,w)
        
        self.getIdx = getIdx
        self.utility = utility
        self.dUtility = dUtility
        self.P = P
        self.g = g
        self.dG = dG
        self.ua = ua
        self.la = la
        self.init = init        
        self.ieqcons = ieqcons
        self.dIeqcons = dIeqcons
       
    def __repr__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)
    
    def __str__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)
        
    def ver(self):
        return 9