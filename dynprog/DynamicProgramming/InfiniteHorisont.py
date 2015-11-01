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




class InfiniterHorisontSolver:
    """
    Implement value interation
    """
    
    def __init__( self, model, interpolator = RegularGridInterpolator ):
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
                        
        

        
        
        self.model = model        
        self.idx = model.idx
        self.interpolator = interpolator
        self.optimisationHistory = []
        #self.doMinimisation = doMinimisationWithGrad if useGrad else doMinimisationWithoutGrad
        
       
    def calculateState( self, pos ):
        return np.array(map( lambda x, i : self.idx[i][x] , 
                            pos, 
                            xrange(len(pos)) ), 
                        dtype = float)
        
    
    
    
        
    
    def testObjectiveFunction( self, w, objFn, s, aStar):
        Aw = self.interpolator( self.idx , w  )
        return objFn(aStar, s, Aw, self)

        
    
    def bellmanOperator(self, w):    
        
        # === Apply linear interpolation to w === #
        Aw = self.interpolator( self.idx , w  )
        
        Tw = np.empty(w.shape)
        sigmas = None 
        messages = np.empty(w.shape, dtype = "O" )
    
    
        it = np.nditer(w, flags=['multi_index'], op_flags=['readonly'])
        while not it.finished:            
            s = self.calculateState(it.multi_index)
            
            
            aStar,f,message = self.model.doMinimisation( s, Aw )
            
            Tw.itemset(it.multi_index,-f)
            messages.itemset(it.multi_index,message)

            if sigmas == None:
                sigmas = [np.empty(w.shape) for i in xrange(len(aStar))]
            
            
            for i in xrange( len(aStar)):
                sigmas[i].itemset(it.multi_index,aStar[i])
            it.iternext()
    
        return ( Tw, sigmas,messages)
        
    #def iterateWithFixedPolicyAndD( self, w, sigmas ):
    #    # === Apply linear interpolation to w === #
    #    Aw = self.interpolator( self.idx , w  )
    #    
    #    Tw = np.empty(w.shape)
    #    dTw = [np.empty(w.shape) for i in xrange(len(sigmas))]
    #
    #    it = np.nditer(w, flags=['multi_index'], op_flags=['readonly'])
    #    while not it.finished:            
    #                   
    #        s = self.calculateState(it.multi_index)
    #        bounds= self.calculateBound(s)       
    #        inits = self.calculateInits(bounds)                        
    #        aStar = map( lambda i:sigmas[i].item(it.multi_index), xrange(len(inits)))            
    #        val, dVal = InfiniterHorisontSolver.objectiveFunctionWithD(aStar, s, Aw, self)
    #        
    #        Tw.itemset(it.multi_index,-val)                        
    #        
    #        for i in xrange(len(sigmas)):
    #            dTw[i].itemset(it.multi_index,-dVal[i])            
    #        
    #        
    #        it.iternext()
    #
    #    return Tw , dTw       
        
    def iterateWithFixedPolicy( self, w, sigmas ):
        # === Apply linear interpolation to w === #
        Aw = RegularGridInterpolator( self.idx , w  )
        
        Tw = np.empty(w.shape)        
    
        it = np.nditer(w, flags=['multi_index'], op_flags=['readonly'])
        while not it.finished:            
                       
            s = self.calculateState(it.multi_index)
            #bounds= self.calculateBound(s)       
            #inits = self.calculateInits(bounds)                        
            aStar = map( lambda i: sigmas[i].item(it.multi_index), xrange(len(sigmas)))            
            val = self.model.objectiveFunction(aStar, s, Aw)
            
            Tw.itemset(it.multi_index,-val)                                   
            
            it.iternext()
    
        return Tw
        
    def computeFixedPointValueIteration(self, w, error_tol=1e-3, max_iter=50, verbose=1):
        iterate = 0
        error = error_tol + 1

        if verbose:
            start_time = time.time()

        while iterate < max_iter and error > error_tol:
            new_w, sigmas, messages = self.bellmanOperator(w)
            iterate += 1
            error = np.max(np.abs(new_w - w))
            w = new_w
            if verbose:
                print 'it:{0}, error:{1}, time from start:{2}\n'.format(iterate, error, time.time() - start_time)    
        

        if verbose:
            print 'Total time {0}'.format(time.time() - start_time)    
        
        return Helpers.Struct( w=w, sigmas=sigmas)
        
    def computeFixedPointPolicyIteration(self, w, error_tol=1e-3, max_iter=50, policy_error_tol=1e-8 , max_policy_iter = 10, verbose=1):
        self.optimisationHistory = []
        
        iterate = 0
        error = error_tol + 1

        if verbose:
            start_time = time.time()

        while iterate < max_iter and error > error_tol:
            new_w, sigmas, messages = self.bellmanOperator(w)
            bellman_new_w = new_w
            
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
            
            self.optimisationHistory+=[Helpers.Struct(bellman_new_w = bellman_new_w,
                                                      new_w = new_w,
                                                      sigmas = sigmas,
                                                      messages = messages,
                                                      iterate = iterate, 
                                                      error = error, 
                                                      timeFromStart = time.time() - start_time, 
                                                      policy_iter = policy_iter, 
                                                      policy_error = policy_error)]
            
            if verbose:
                print 'it:{0}, error:{1}, time from start:{2}, policy_iter:{3}, policy_error:{4}\n'.format(iterate, error, time.time() - start_time, policy_iter, policy_error)    
                
        

        if verbose:
            print 'Total time {0}'.format(time.time() - start_time)    
        
        return Helpers.Struct( w=w, sigmas=sigmas)
    
    def ver():
        return 1
        
