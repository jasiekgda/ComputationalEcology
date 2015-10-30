# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 16:33:57 2015

@author: root
"""
import numpy as np
import scipy as sp
import scipy.optimize as spo
from numba import jit


class DynamicModel:
    """
    class provide:
    
    utility: callable( s:vector, a:vector ), state utility function, 
    g: callable( s:vector, a:vector, e:vector) - state transition,
    la: callable( s:vector )
    ua: callable( s:vector )
    beta: scalar    
    mu: [ (e,w)] stochastic changes, w - waight, e - stochastic update
    s: state
    a: action
    e: stochastic update
    
    
    Parameters
    ----------
    utility
    g
    beta
    w
    """
    def __init__(self, utility, dutility, g, dg, la, ua, beta=0.95, mu = [(0,1)]):
        self.utility, self.dutility, self.g, self.dg = jit(utility), jit(dutility), jit(g), jit(dg)
        self.ls, self.us = jit(la), jit( ua )
        self.beta, self.w  = beta, mu
        
        np.vectorize
        
    def __repr__(self):
        return "xx"
    
    def __str__(self):
        return "xx"
        
class ValueIterationSolver:
    """
    Implement value interation
    """
    
    def __init__( self, model, grid ):
        """
        model:  DynamicModel, model definition
        grid: np.mgrid, interpolation points
        """
        self.model = model        
        self.grid = grid
        
        utility, dutility, g, dg , beta, w = model.utility, model.dutility, model.g, model.dg, model.beta, model.w      
        
        self.bellmanFunction = lambda a , s, vInter: utility(s) + beta*np.sum([ v(g(s,c,e)) for (e,w) in mu]
        #self.bellmanFunction = jit(self.bellmanFunction)
        
    def bellmanOperator(self, v):
        """
        The approximate Bellman operator, which computes and returns the
        updated value function Tw on the grid points.

        Parameters
        ----------
        v : array_like(float, ndim=1)
            The value of the input function on different grid points
        compute_policy : Boolean, optional(default=False)
            Whether or not to compute policy function

        """
        # === Apply linear interpolation to w === #
        Aw = lambda x: interp(x, self.grid, w)

        if compute_policy:
            sigma = np.empty(len(w))

        # == set Tw[i] equal to max_c { u(c) + beta w(f(k_i) - c)} == #
        Tw = np.empty(len(w))
        for i, k in enumerate(self.grid):
            objective = lambda c: - self.u(c) - self.beta * Aw(self.f(k) - c)
            c_star = fminbound(objective, 1e-6, self.f(k))
            if compute_policy:
                # sigma[i] = argmax_c { u(c) + beta w(f(k_i) - c)}
                sigma[i] = c_star
            Tw[i] = - objective(c_star)

        if compute_policy:
            return Tw, sigma
        else:
            return Tw