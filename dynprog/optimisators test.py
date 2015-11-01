# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 17:40:45 2015

@author: root
"""

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import fmin_slsqp

def func(x, sign=1.0):
    """ Objective function """
    return sign*(2*x[0]*x[1] + 2*x[0] - x[0]**2 - 2*x[1]**2)
    
def func_deriv(x, sign=1.0):
    """ Derivative of objective function """
    dfdx0 = sign*(-2*x[0] + 2*x[1] + 2)
    dfdx1 = sign*(2*x[0] - 4*x[1])
    return np.array([ dfdx0, dfdx1 ])
    
cons = ({'type': 'eq',
         'fun' : lambda x: np.array([x[0]**3 - x[1]]),
         'jac' : lambda x: np.array([3.0*(x[0]**2.0), -1.0])},
        {'type': 'ineq',
         'fun' : lambda x: np.array([x[1] - 1]),
         'jac' : lambda x: np.array([0.0, 1.0])})
         
res = minimize(func, [-1.0,1.0], args=(-1.0,), fprime=func_deriv,
               f_ieqcons,
               fprime_f_ieqcons
               
               )





print(res.x)

def func( x ):
    (x[0]-1)**2 + (x[1]-1)**2

def fprime( x ):    
    return np.array([2*(x[0]-1), 2*(x[1]-1)])

def f_ieqcons( x ):
    return 1.0 - x[0] - x[1]

def fprime_ieqcons ( x ):
    return np.array([[-1,-1]])

fmin_slsqp(func func , 
           x0 = [-1.0,1.0], 
           bounds = [(0,1),(0,1)],
           fprime=fprime,
           f_ieqcons = f_ieqcons,
           fprime_ieqcons  = fprime_ieqcons )
