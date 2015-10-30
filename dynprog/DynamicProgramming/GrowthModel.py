# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 18:26:24 2015

@author: root
"""
import numpy as np
from numpy import log

class GrowthModel:
    alpha = 0.65
    beta = 0.95       
    
    @staticmethod
    def getIdx(grid_max = 2 , grid_size = 150):
        return [ np.linspace(1e-6, grid_max, grid_size) ]
    
    @staticmethod
    def utility( s, a ):
        return log(a)

    @staticmethod
    def g( s, a, e):
        #ret = min(max(s[0]**GrowthModel.alpha - a[0],2e-6),1.99)    
        ret = s[0]**GrowthModel.alpha - a[0]
        return np.array([ret])

    @staticmethod
    def ua( s ):
        return s**GrowthModel.alpha

    @staticmethod
    def la( s ):
        return np.array([1e-6],dtype = float)
        
    @staticmethod
    def init( idx ):
        return 5*log( idx[0])-25
        
    @staticmethod
    def vStar(idx):
        ab = GrowthModel.alpha * GrowthModel.beta
        c1 = (log(1 - ab) + log(ab) * ab / (1 - ab)) / (1 - GrowthModel.beta)
        c2 = GrowthModel.alpha / (1 - ab)
        return c1 + c2*log(idx)        
