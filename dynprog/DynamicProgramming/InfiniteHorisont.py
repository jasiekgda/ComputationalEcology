# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 22:02:55 2015

@author: root
"""

from __future__ import division  # Omit for Python 3.x
import numpy as np
import scipy.optimize as spo
from scipy.interpolate import RegularGridInterpolator
from numba import jit
import Helpers
import time


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
    def __init__(self, utility, dUtility, g, dG, la, ua, idx , beta=0.95, mu = [(0,1)]):
        self.utility = jit(utility)  
        self.dUtility = jit(dUtility)
        self.dG = jit(dG)
        self.la, self.ua = jit(la), jit( ua )
        self.beta, self.mu  = beta, mu
        self.idx = idx 
        self.sMax = np.array(map( lambda x: max(x), idx), dtype = float )
        self.sMin = np.array(map( lambda x: min(x), idx), dtype = float )

        gjit = jit(g)

        def gWithSClipped( s, a, e ):
            sp = gjit( s, a, e )
            np.clip( sp, self.sMin , self.sMax, out = sp)
            return sp       
        
        self.g = jit(gWithSClipped)
        
    def __repr__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)
    
    def __str__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)


class InfiniterHorisontSolver:
    """
    Implement value interation
    """
    
    def __init__( self, model, interpolator = RegularGridInterpolator, useGrad = False ):
        """
        model:  DynamicModel, model definition
        idx: indexes along axis
        """
        def doMinimisationWithoutGradTNC( inits, bounds, args ):
            return spo.fmin_tnc(InfiniterHorisontSolver.objectiveFunction, 
                                inits ,
                                bounds=bounds,
                                approx_grad = True, 
                                maxCGit = 0,
                                disp = 0,
                                args = args )[0]
                        
        def doMinimisationWithGradTNC( inits, bounds, args ):
            return spo.fmin_tnc(InfiniterHorisontSolver.objectiveFunctionWithD,
                                inits ,
                                bounds=bounds,   
                                maxCGit = 0,
                                disp = 0,
                                args = args )[0]
        
        
        def doMinimisationWithoutGrad( bounds, args ):
            inits = self.calculateInits(bounds)
            return spo.fmin_l_bfgs_b(InfiniterHorisontSolver.objectiveFunction, 
                                inits ,
                                bounds=bounds,
                                approx_grad = True, 
                                disp = 0,
                                args = args )[0]
                        
        def doMinimisationWithGrad(  bounds, args ):
            inits = self.calculateInits(bounds)
            return spo.fmin_l_bfgs_b(InfiniterHorisontSolver.objectiveFunctionWithD,
                                inits ,
                                bounds=bounds,                                  
                                disp = 0,
                                args = args )[0]

        
        
        self.model = model        
        self.idx = model.idx
        self.interpolator = interpolator
        self.doMinimisation = doMinimisationWithGrad if useGrad else doMinimisationWithoutGrad
        
       
    @staticmethod
    def objectiveFunction( a, s, Aw , solver):
        return -( solver.model.utility(s,a) + solver.model.beta * sum( [w*Aw(solver.model.g(s,a,e)) for e, w in solver.model.mu]))
        
    @staticmethod
    def objectiveFunctionWithD( a, s, Aw, solver):
        
        val = solver.model.utility(s,a)
        dVal = solver.model.dUtility(s,a)        
        
        for e, w in solver.model.mu:
            FdF = Aw.FdF(solver.model.g(s,a,e))
            #print "F:", FdF[0], " dF:",FdF[1]
            val += solver.model.beta * w * FdF[0]
            dVal += solver.model.beta * w * np.dot(FdF[1], solver.model.dG(s,a,e))            
        
        return (-val,-dVal)
    
    def calculateState( self, pos ):
        return np.array(map( lambda x, i : self.idx[i][x] , 
                            pos, 
                            xrange(len(pos)) ), 
                        dtype = float)
        
    
    
    def calculateBound( self, s):
        return zip( self.model.la(s), self.model.ua(s))
    
   
    def calculateInits( self, bounds):
        return map(lambda x: (0.2*x[0]+0.8*x[1]), bounds)        
        
    
    def testObjectiveFunction( self, w, objFn, s, aStar):
        Aw = self.interpolator( self.idx , w  )
        return objFn(aStar, s, Aw, self)

        
    
    def bellmanOperator(self, w):    
        
        # === Apply linear interpolation to w === #
        Aw = self.interpolator( self.idx , w  )
        
        Tw = np.empty(w.shape)
        sigmas = None 
    
    
        it = np.nditer(w, flags=['multi_index'], op_flags=['readonly'])
        while not it.finished:            
            s = self.calculateState(it.multi_index)
            bounds= self.calculateBound(s)       
            
            aStar = self.doMinimisation( bounds , (s, Aw, self) )
            
            Tw.itemset(it.multi_index,-InfiniterHorisontSolver.objectiveFunction(aStar, s, Aw, self))

            if sigmas == None:
                sigmas = [np.empty(w.shape) for i in xrange(len(bounds))]
            
            
            for i in xrange( len(bounds)):
                sigmas[i].itemset(it.multi_index,aStar[i])
            it.iternext()
    
        return ( Tw, sigmas)
        
    def iterateWithFixedPolicyAndD( self, w, sigmas ):
        # === Apply linear interpolation to w === #
        Aw = self.interpolator( self.idx , w  )
        
        Tw = np.empty(w.shape)
        dTw = [np.empty(w.shape) for i in xrange(len(sigmas))]
    
        it = np.nditer(w, flags=['multi_index'], op_flags=['readonly'])
        while not it.finished:            
                       
            s = self.calculateState(it.multi_index)
            bounds= self.calculateBound(s)       
            inits = self.calculateInits(bounds)                        
            aStar = map( lambda i:sigmas[i].item(it.multi_index), xrange(len(inits)))            
            val, dVal = InfiniterHorisontSolver.objectiveFunctionWithD(aStar, s, Aw, self)
            
            Tw.itemset(it.multi_index,-val)                        
            
            for i in xrange(len(sigmas)):
                dTw[i].itemset(it.multi_index,-dVal[i])            
            
            
            it.iternext()
    
        return Tw , dTw       
        
    def iterateWithFixedPolicy( self, w, sigmas ):
        # === Apply linear interpolation to w === #
        Aw = RegularGridInterpolator( self.idx , w  )
        
        Tw = np.empty(w.shape)        
    
        it = np.nditer(w, flags=['multi_index'], op_flags=['readonly'])
        while not it.finished:            
                       
            s = self.calculateState(it.multi_index)
            bounds= self.calculateBound(s)       
            inits = self.calculateInits(bounds)                        
            aStar = map( lambda i:sigmas[i].item(it.multi_index), xrange(len(inits)))            
            val = InfiniterHorisontSolver.objectiveFunction(aStar, s, Aw, self)
            
            Tw.itemset(it.multi_index,-val)                                   
            
            it.iternext()
    
        return Tw
        
    def computeFixedPointValueIteration(self, w, error_tol=1e-3, max_iter=50, verbose=1):
        iterate = 0
        error = error_tol + 1

        if verbose:
            start_time = time.time()

        while iterate < max_iter and error > error_tol:
            new_w, sigmas = self.bellmanOperator(w)
            iterate += 1
            error = np.max(np.abs(new_w - w))
            w = new_w
            if verbose:
                print 'it:{0}, error:{1}, time from start:{2}\n'.format(iterate, error, time.time() - start_time)    
        

        if verbose:
            print 'Total time {0}'.format(time.time() - start_time)    
        
        return Helpers.Struct( w=w, sigmas=sigmas)
        
    def computeFixedPointPolicyIteration(self, w, error_tol=1e-3, max_iter=50, policy_error_tol=1e-8 , max_policy_iter = 10, verbose=1):
        iterate = 0
        error = error_tol + 1

        if verbose:
            start_time = time.time()

        while iterate < max_iter and error > error_tol:
            new_w, sigmas = self.bellmanOperator(w)
            
            policy_iter = 0
            policy_error = policy_error_tol + 1
            while policy_iter < max_policy_iter and policy_error > policy_error_tol:
                policy_new_w = self.iterateWithFixedPolicy( new_w, sigmas )
                policy_iter = policy_iter + 1
                policy_error = np.max(np.abs(policy_new_w - new_w))
                new_w = policy_new_w
                
            
                
            
            iterate += 1
            error = np.max(np.abs(new_w - w))
            w = new_w
            if verbose:
                print 'it:{0}, error:{1}, time from start:{2}, policy_iter:{3}, policy_error:{4}\n'.format(iterate, error, time.time() - start_time, policy_iter, policy_error)    
                
        

        if verbose:
            print 'Total time {0}'.format(time.time() - start_time)    
        
        return Helpers.Struct( w=w, sigmas=sigmas)
    
    def ver():
        return 1
        

class PolicyIterationSolver:
    """
    Implement interation policy solver
    """
    def __init__( self, model, idx ):
        """
        model:  DynamicModel, model definition
        idx: indexes along axis
        """
        self.model = model        
        self.idx = idx