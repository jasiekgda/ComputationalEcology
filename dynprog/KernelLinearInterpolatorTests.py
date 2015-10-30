# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 18:40:15 2015

@author: root
"""

from DynamicProgramming.Helpers import *
from DynamicProgramming.Interpolators import *

#idx = [ np.linspace(0,1,11), np.linspace(2,3,11)]
#def fn( x,y ):
#    return x+10*y
#Z = MathHelpers.EvalFunctionAlongAxis( idx , fn )
#GraphicsHelpers.plot3D( idx , Z , xlabel = "idx 1" , ylabel = "idx 2" , zlabel = "value" )

#intp = KernelLinearInterpolator([np.array([0,1]),np.array([1,2])],np.array([[1,1.5],[2,2.5]]))
#idx = [ np.linspace(0,1,21), np.linspace(1,2,21)]
#GraphicsHelpers.plot3D( idx , MathHelpers.EvalFunctionAlongAxis( idx , lambda x,y: intp(np.array([x,y])) ) , xlabel = "idx 1" , ylabel = "idx 2" , zlabel = "value" )



intp = KernelLinearInterpolator([np.array([0,1,2]),np.array([0,1,2])],np.array([[0,0,0],[0,2,0],[0,0,0]]))
idx = [ np.linspace(0,2,21), np.linspace(0,2,21)]
GraphicsHelpers.plot3D( idx , MathHelpers.EvalFunctionAlongAxis( idx , lambda x,y: intp(np.array([x,y])) ) , xlabel = "idx 1" , ylabel = "idx 2" , zlabel = "value" )
GraphicsHelpers.plot3D( idx , MathHelpers.EvalFunctionAlongAxis( idx , lambda x,y: intp.FdF(np.array([x,y]))[0] ) , xlabel = "idx 1" , ylabel = "idx 2" , zlabel = "value" )
GraphicsHelpers.plot3D( idx , MathHelpers.EvalFunctionAlongAxis( idx , lambda x,y: intp.FdF(np.array([x,y]))[1][0] ) , xlabel = "idx 1" , ylabel = "idx 2" , zlabel = "value" )
GraphicsHelpers.plot3D( idx , MathHelpers.EvalFunctionAlongAxis( idx , lambda x,y: intp.FdF(np.array([x,y]))[1][1] ) , xlabel = "idx 1" , ylabel = "idx 2" , zlabel = "value" )



intp = KernelLinearInterpolator([np.array([0,1,2,3,4]),np.array([0,1,2,3,4])],np.array([[0,0,0,0,0],[0,0,0,0,0],[0,0,2,0,0],[0,0,0,0,0],[0,0,0,0,0]]))
idx = [ np.linspace(0,4,9), np.linspace(0,4,9)]
Z = MathHelpers.EvalFunctionAlongAxis( idx , lambda x,y: intp(np.array([x,y])) )
GraphicsHelpers.plot3D( idx , Z , xlabel = "idx 1" , ylabel = "idx 2" , zlabel = "value" )

intp = KernelLinearInterpolator([np.array([0,1,2,3,4]),np.array([0,1,2,3,4])],np.array([[1,0,0,0,0],[0,0,0,0,0],[0,0,2,0,0],[0,0,0,0,0],[0,0,0,0,1]]))
idx = [ np.linspace(0,4,9), np.linspace(0,4,9)]
Z = MathHelpers.EvalFunctionAlongAxis( idx , lambda x,y: intp(np.array([x,y])) )
GraphicsHelpers.plot3D( idx , Z , xlabel = "idx 1" , ylabel = "idx 2" , zlabel = "value" )



intp = KernelLinearInterpolator([np.array([0,1]),np.array([1,2])],np.array([[1,1],[1,1]]))
idx = [ np.linspace(0,1,5), np.linspace(1,2,5)]
Z = MathHelpers.EvalFunctionAlongAxis( idx , lambda x,y: intp(np.array([x,y])) )
GraphicsHelpers.plot3D( idx , Z , xlabel = "idx 1" , ylabel = "idx 2" , zlabel = "value" )
Z


intp = GridLinearInterpolator([np.array([0,1]),np.array([1,2])],np.array([[1,1],[1,1]]))
idx = [ np.linspace(0,1,5), np.linspace(1,2,5)]
Z = MathHelpers.EvalFunctionAlongAxis( idx , lambda x,y: intp(np.array([x,y])) )
GraphicsHelpers.plot3D( idx , Z , xlabel = "idx 1" , ylabel = "idx 2" , zlabel = "value" )
Z

intp = GridLinearInterpolator([np.array([0,1,2,3,4]),np.array([0,1,2,3,4])],np.array([[1,0,0,0,0],[0,0,0,0,0],[0,0,2,0,0],[0,0,0,0,0],[0,0,0,0,1]]))
idx = [ np.linspace(0,4,9), np.linspace(0,4,9)]
Z = MathHelpers.EvalFunctionAlongAxis( idx , lambda x,y: intp(np.array([x,y])) )
GraphicsHelpers.plot3D( idx , Z , xlabel = "idx 1" , ylabel = "idx 2" , zlabel = "value" )

from DynamicProgramming.Helpers import *
from DynamicProgramming.Interpolators import *

intp = GridLinearInterpolator([np.array([0,1]),np.array([1,2])],np.array([[1,1],[1,1]]))
idx = [ np.linspace(0,1,5), np.linspace(1,2,5)]
Z = MathHelpers.EvalFunctionAlongAxis( idx , lambda x,y: intp(np.array([x,y])) )
GraphicsHelpers.plot3D( idx , Z , xlabel = "idx 1" , ylabel = "idx 2" , zlabel = "value" )
Z

intp = GridLinearInterpolator([np.array([0,1,2]),np.array([0,1,2])],np.array([[0,0,0],[0,1,0],[0,0,0]]))
idx = [ np.linspace(0,2,9), np.linspace(0,2,9)]
Z = MathHelpers.EvalFunctionAlongAxis( idx , lambda x,y: intp(np.array([x,y])) )
GraphicsHelpers.plot3D( idx , Z , xlabel = "idx 1" , ylabel = "idx 2" , zlabel = "value" )
Z

intp = GridLinearInterpolator([np.array([0,1,2,3,4]),np.array([0,1,2,3,4])],np.array([[0,0,0,0,0],[0,0,0,0,0],[0,0,1,0,0],[0,0,0,0,0],[0,0,0,0,0]]))
idx = [ np.linspace(0,4,33), np.linspace(0,4,33)]
Z = MathHelpers.EvalFunctionAlongAxis( idx , lambda x,y: intp.FdF(np.array([x,y]))[0] )
ZdX = MathHelpers.EvalFunctionAlongAxis( idx , lambda x,y: intp.FdF(np.array([x,y]))[1][0] )
ZdY = MathHelpers.EvalFunctionAlongAxis( idx , lambda x,y: intp.FdF(np.array([x,y]))[1][1] )

GraphicsHelpers.plot3D( idx , Z , xlabel = "idx 1" , ylabel = "idx 2" , zlabel = "value" )
GraphicsHelpers.plot3D( idx , ZdX , xlabel = "idx 1" , ylabel = "idx 2" , zlabel = "value" )
GraphicsHelpers.plot3D( idx , ZdY , xlabel = "idx 1" , ylabel = "idx 2" , zlabel = "value" )

intp = GridLinearInterpolator([np.array([0,1,2,3,4]),np.array([0,1,2,3,4])],np.array([[1,0,0,0,0],[0,0,0,0,0],[0,0,2,0,0],[0,0,0,0,0],[0,0,0,0,1]]))
idx = [ np.linspace(0,4,9), np.linspace(0,4,9)]
Z = MathHelpers.EvalFunctionAlongAxis( idx , lambda x,y: intp(np.array([x,y])) )
GraphicsHelpers.plot3D( idx , Z , xlabel = "idx 1" , ylabel = "idx 2" , zlabel = "value" )

from DynamicProgramming.Interpolators import *
from DynamicProgramming.Helpers import *


intp = GridLinearInterpolator([np.array([0,1]),np.array([0,1])],np.array([[0,1],[2,3]]))
#idx = [ np.linspace(-1,2,13), np.linspace(-1,2,13)]
idx = [ np.linspace(0,1,11), np.linspace(0,1,11)]
Z = MathHelpers.EvalFunctionAlongAxis( idx , lambda x,y: intp.FdF(np.array([x,y]))[0] )
ZdX = MathHelpers.EvalFunctionAlongAxis( idx , lambda x,y: intp.FdF(np.array([x,y]))[1][0] )
ZdY = MathHelpers.EvalFunctionAlongAxis( idx , lambda x,y: intp.FdF(np.array([x,y]))[1][1] )
GraphicsHelpers.plot3D( idx , Z , xlabel = "idx 1" , ylabel = "idx 2" , zlabel = "value" )
GraphicsHelpers.plot3D( idx , ZdX , xlabel = "idx 1" , ylabel = "idx 2" , zlabel = "value" )
GraphicsHelpers.plot3D( idx , ZdY , xlabel = "idx 1" , ylabel = "idx 2" , zlabel = "value" )



import scipy.optimize as spo

def fn(x):
    print x
    return x[0]**2 + 0.001*x[1]**2
    
spo.fmin_tnc( fn , [1,1], bounds = [[-1,1],[-1,1]],approx_grad = True )