# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 18:14:59 2015

@author: root
"""
import numpy as np
from itertools import product
from operator import mul
from numba import jit
from sys import float_info


class BaseInterpolator(object):
    def __init__(self, points, values):
        self.points = points;
        self.values = values;
        self.dim = np.array( map(lambda x: x.shape[0], points ) , dtype = int )
        self.minDeltas = np.array( map( lambda x: np.min(np.diff(x)) , points ) , dtype = float )

    @jit
    def find_indices(self, xi):
        size = len(xi)
        # find relevant edges between which xi are situated
        indices = np.ndarray([size], dtype = float)
        # compute distance to lower edge in unity units
        norm_distances = np.ndarray([size], dtype = float)
        delta = np.ndarray([size], dtype = float)      
        
        # iterate through dimensions
        for x, grid, pos in zip(xi, self.points, xrange(size)):
            i = np.searchsorted(grid, x) - 1
            if i < 0:
                i = 0
            if i > grid.size - 2:
                i = grid.size - 2
            indices[pos] = i
            norm_distances[pos] = ((x - grid[i]) /
                                  (grid[i + 1] - grid[i]))  
            delta[pos]=(grid[i + 1] - grid[i])
            
        return indices, norm_distances, delta
        
    def __call__(self,xi):
        pass
    
    def FdF(self,xi):
        pass
        

class KernelLinearInterpolator(BaseInterpolator):

    def __init__(self,points, values):
        super(KernelLinearInterpolator, self).__init__(points, values)

    def __call__(self,xi):
        #idxs = np.array( map(lambda i: self.points[i].searchsorted(xi[i],side='right')-1,xrange(xi.shape[0])),dtype = int)
        indices, norm_distances, delta = self.find_indices(xi)

        edges = product(*[[i, i + 1] for i in indices])       
        
        value = 0.
        for edge_indices in edges:
            dist = [yi if ei == i else 1-yi for ei, i, yi in zip(edge_indices, indices, norm_distances)]                                       
            weight = max(1-max(dist),0)            
            value += self.values[edge_indices] * weight            
        
        return value
        
        #if any(idxs<0) or any( idxs >= self.dim ): 
        #    raise Exception('xi out of range')
            
        #return [self.valueFor(idxs+delta,xi )  for delta in product([0,1],repeat = len(idxs))]

    def FdF(self,xi):

        
        indices, norm_distances, delta = self.find_indices(xi)

        edges = product(*[[i, i + 1] for i in indices])       
        
        value = 0.
        dF = np.zeros( xi.shape, dtype = float )
        
        for edge_indices in edges:
            dist = [yi if ei == i else 1-yi for ei, i, yi in zip(edge_indices, indices, norm_distances)]
            weight = max(1-max(dist),0)            
            value += self.values[edge_indices] * weight            

            grad = np.array([1.0 if ei == i else -1.0 for ei, i, yi in zip(edge_indices, indices, norm_distances)])
            dF += self.values[edge_indices]*grad
        
        return (value, dF)
        
        
        
        
class GridLinearInterpolator(BaseInterpolator):
    
    def __init__(self,points, values):
        super(GridLinearInterpolator, self).__init__(points, values)    
    
    def __call__(self,xi):    
        indices, norm_distances, delta = self.find_indices(xi)        
        
        edges = product(*[[i, i + 1] for i in indices])
        value = 0.
        for edge_indices in edges:
            revdist = [1-yi if ei == i else yi for ei, i, yi in zip(edge_indices, indices, norm_distances)]            
            weight = reduce( mul, revdist, 1)
            value += self.values[edge_indices] * weight            
        return value
        
    def FdF( self, xi):       
        
        val = self(xi)
        dVal = np.ndarray(xi.shape,dtype = float)
        
        for i in xrange(xi.shape[0]):
            epsilon = self.minDeltas[i]/1000.0
            xi[i] += epsilon
            dVal[i] = self(xi)
            xi[i] -= 2.0*epsilon
            dVal[i] -= self(xi)            
            xi[i] += epsilon
            dVal[i] /= 2.0*epsilon
            
        
        return (val, dVal)       
        
            
        
    
    def FdFexact(self,xi):        
        indices, norm_distances, delta = self.find_indices(xi)

        edges = product(*[[i, i + 1] for i in indices])       
        
        value = 0.
        dF = np.zeros( xi.shape, dtype = float )
        
        for edge_indices in edges:
            revdist = np.array([1-yi if ei == i else yi for ei, i, yi in zip(edge_indices, indices, norm_distances)],dtype = float)
            weight = np.prod(revdist)
            value += self.values[edge_indices] * weight            

            grad = np.array([-1.0 if ei == i else 1.0 for ei, i, yi in zip(edge_indices, indices, norm_distances)])/delta
            
            if weight != 0.0 and weight != 1.0:
                dF += self.values[edge_indices]*weight*grad/revdist    
        
        return (value, dF)