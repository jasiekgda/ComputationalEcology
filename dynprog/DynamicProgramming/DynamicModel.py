# -*- coding: utf-8 -*-
"""
Created on Sun Nov 01 21:01:38 2015

@author: root
"""

from __future__ import division  # Omit for Python 3.x
import numpy as np
import scipy.optimize as spo
from numba import jit

 
class DynamicModelBase(object):
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
    def __init__(self, utility, g, la, ua, idx , beta=0.95, mu = [(0,1)]):
        self.utility = jit(utility)  
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
        
    
    
    
    def objectiveFunction( self, a, s, Aw ):
        return -( self.utility(s,a) + self.beta * sum( [w*Aw(self.g(s,a,e)) for e, w in self.mu]))        
        
    def __repr__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)
    
    def __str__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)
        
    def lastOptimisationCode(self):
        return 0


class DynamicModelWithGradOptimisation(DynamicModelBase):
    '''
    use l_bfgs_b with gradiant
    '''
    def __init__(self, utility, dUtility, g, dG, la, ua, idx , beta=0.95, mu = [(0,1)]):
        
        super(DynamicModelWithGradOptimisation, self).__init__(utility, g, la, ua, idx , beta, mu )
        self.dUtility = jit(dUtility)
        self.dG = jit(dG)
        
    
    
    @staticmethod
    def objectiveFunctionWithGrad( a, s, Aw, model):
        
        val = model.utility(s,a)
        dVal = model.dUtility(s,a)        
        
        for e, w in model.mu:
            FdF = Aw.FdF(model.g(s,a,e))
            #print "F:", FdF[0], " dF:",FdF[1]
            val += model.beta * w * FdF[0]
            dVal += model.beta * w * np.dot(FdF[1], model.dG(s,a,e))            
        
        return (-val,-dVal)
        
    def calculateBound( self, s):
        return zip( self.la(s), self.ua(s))
    
   
    def calculateInits( self, bounds):
        return map(lambda x: (0.2*x[0]+0.8*x[1]), bounds)        
    
    def doMinimisation(  self, s , Aw ):
            bounds= self.calculateBound(s)               
            inits = self.calculateInits(bounds)
            ret = spo.fmin_l_bfgs_b(DynamicModelWithGradOptimisation.objectiveFunctionWithGrad,
                                inits ,
                                bounds=bounds,                                  
                                disp = 0,
                                args = (s, Aw, self) )            
            
            return ret
            
    @staticmethod
    def printErrors( optimisationHistory ):
        errorsCnt = 0
        for i in xrange(len( optimisationHistory )):
            it = np.nditer(optimisationHistory[i].messages, flags=['multi_index','refs_ok'], op_flags=['readonly'])
            
            while not it.finished:            

                msg = optimisationHistory[i].messages.item(it.multi_index)
                if msg['warnflag'] == 1:
                    print "iteration:",i," index ",it.multi_index , " flag: 1 (to many evaluations)"
                    errorsCnt += 1
                if msg['warnflag'] == 2:
                    print "iteration:",i," index ",it.multi_index , " flag: 2 ", msg['task']
                    errorsCnt += 1
                
                it.iternext()
                
        if errorsCnt == 0:
            print "no errors"
            
class DynamicModelSLSQP(DynamicModelBase):
    '''
    use l_bfgs_b with gradiant
    '''
    def __init__(self, utility, dUtility, g, dG, ieqcons, dIeqcons, la, ua, idx , beta=0.95, mu = [(0,1)]):
        
        super(DynamicModelSLSQP, self).__init__(utility, g, la, ua, idx , beta, mu )
        self.dUtility = jit(dUtility)
        self.dG = jit(dG)
        self.ieqcons = jit(ieqcons)
        self.dIeqcons = jit(dIeqcons)
        
    
    
    @staticmethod
    def objectiveFunctionWithGrad( a, s, Aw, model):
        
        val = model.utility(s,a)
        dVal = model.dUtility(s,a)        
        
        for e, w in model.mu:
            FdF = Aw.FdF(model.g(s,a,e))
            #print "F:", FdF[0], " dF:",FdF[1]
            val += model.beta * w * FdF[0]
            dVal += model.beta * w * np.dot(FdF[1], model.dG(s,a,e))            
        
        return (-val,-dVal)
        
    def calculateBound( self, s):
        return zip( self.la(s), self.ua(s))
    
   
    def calculateInits( self, bounds):
        return map(lambda x: (0.2*x[0]+0.8*x[1]), bounds)        
    
    @staticmethod
    def func( a, *args):
        s, Aw, model = args
        val = model.utility(s,a)
        dVal = model.dUtility(s,a)        
        
        for e, w in model.mu:
            FdF = Aw.FdF(model.g(s,a,e))
            #print "F:", FdF[0], " dF:",FdF[1]
            val += model.beta * w * FdF[0]
            dVal += model.beta * w * np.dot(FdF[1], model.dG(s,a,e))            
        
        return (-val)    
    
    @staticmethod
    def fprime( a, *args):
        s, Aw, model = args
        val = model.utility(s,a)
        dVal = model.dUtility(s,a)        
        
        for e, w in model.mu:
            FdF = Aw.FdF(model.g(s,a,e))
            #print "F:", FdF[0], " dF:",FdF[1]
            val += model.beta * w * FdF[0]
            dVal += model.beta * w * np.dot(FdF[1], model.dG(s,a,e))            
        
        return (-dVal)        
    
    @staticmethod
    def _ieqcons( a, *args):
        s, Aw, model = args
        return model.ieqcons(s,a)
        
    @staticmethod
    def _dIeqcons( a, *args):
        s, Aw, model = args
        return model.dIeqcons(s,a)
        
    
    def doMinimisation(  self, s , Aw ):
        bounds= self.calculateBound(s)               
        inits = self.calculateInits(bounds)
                    
        ret = spo.fmin_slsqp(func = DynamicModelSLSQP.func, 
                             x0 = inits, 
                             bounds = bounds,
                             fprime=DynamicModelSLSQP.fprime,
                             f_ieqcons = DynamicModelSLSQP._ieqcons,
                             fprime_ieqcons  = DynamicModelSLSQP._dIeqcons,
                             full_output = True,
                             disp = 0 ,
                             args = (s, Aw, self) )            
            
        return (ret[0],ret[1],( ret[2], ret[3], ret[4]))
            
    @staticmethod
    def printErrors( optimisationHistory ):
        errorsCnt = 0
        for i in xrange(len( optimisationHistory )):
            it = np.nditer(optimisationHistory[i].messages, flags=['multi_index','refs_ok'], op_flags=['readonly'])
            
            while not it.finished:
                msg = optimisationHistory[i].messages.item(it.multi_index)
                if msg[1] != 0:
                    print "iteration:",i," index ",it.multi_index , " error: ", msg[2]
                    errorsCnt += 1
                
                
                it.iternext()
                
        if errorsCnt == 0:
            print "no errors"
        