# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 14:07:33 2015

@author: root
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

class DynamicsSymulation:
    
    def __init__(self, model, policy ):
        self.model = model
        self.policy = policy        
        self.mu = np.asarray(model.mu)
        self.bounds = self.mu[:,1].cumsum()
        
        self.interpolators = [ RegularGridInterpolator(model.idx,p) for p in policy]
      
    def generateEvents( self, length ):
        vals = np.random.uniform(size = length)
        return self.mu[np.searchsorted(self.bounds, vals),0]

    
    def calcPolicy( self, state ):
        return np.array([interp(state)[0] for interp in self.interpolators], dtype = float)
    
     
    def simulateTrial( self, length,  initState):
        
        state = np.ndarray( [length+1,initState.shape[0]], dtype = float )
        action = np.ndarray( [length, len(self.policy)], dtype = float)
        state[0,:]=initState
        
        
        events = self.generateEvents(length)
        
        
        for i in xrange(1, length+1):
            currState = state[i-1,:]
            action[i-1,:] = self.calcPolicy(currState)
            state[i,:] = self.model.g(currState,action[i-1,:],events[i-1])           
             
        
        return ( state, events, action)
        
    