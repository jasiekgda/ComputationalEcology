# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 19:34:07 2015

@author: root
"""



from DynamicProgramming.Interpolators import *
from DynamicProgramming.Helpers import *
from DynamicProgramming.InfiniteHorisont import *
from DynamicProgramming.ProductionInventoryModel import *


productionInventoryModel = ProductionInventoryModel()
idx = productionInventoryModel.getIdx(grid_size = [4,20])

w0 = productionInventoryModel.init(idx)
model = DynamicModel(productionInventoryModel.utility,
                     productionInventoryModel.dUtility,
                     productionInventoryModel.g,
                     productionInventoryModel.dG,
                     productionInventoryModel.la,
                     productionInventoryModel.ua,
                     idx,
                     productionInventoryModel.beta,
                     productionInventoryModel.weigths(n=3))

solverWithDiff = InfiniterHorisontSolver(model, interpolator = GridLinearInterpolator, useGrad = True)
retDiff = solverWithDiff.computeFixedPointPolicyIteration(w0,max_iter = 10,policy_iter = 100)

GraphicsHelpers.plot3D( idx , retDiff.w.T , xlabel = "stock" , ylabel = "price" , zlabel = "value" )
GraphicsHelpers.plot3D( idx , retDiff.sigmas[0].T , xlabel = "stock" , ylabel = "price" , zlabel = "produce" )
GraphicsHelpers.plot3D( idx , retDiff.sigmas[1].T , xlabel = "stock" , ylabel = "price" , zlabel = "store" )

#retDiff.sigmas[1]
#retDiff.sigmas[0]

ret2it,dRet2it = solverWithDiff.iterateWithFixedPolicyAndD( retDiff.w, retDiff.sigmas )
retDiff.sigmas[1][:,:] = 0
ret2itSigmaFixed, dRet2itSigmaFixed = solverWithDiff.iterateWithFixedPolicyAndD( retDiff.w, retDiff.sigmas )
GraphicsHelpers.plot3D( idx , (ret2it-ret2itSigmaFixed).T , xlabel = "stock" , ylabel = "price" , zlabel = "value" )


dRet2it[1]

#solver.testObjectiveFunction(w0,InfiniterHorisontSolver.objectiveFunction,np.array([1,1]),np.array([1,1]))
#solver.testObjectiveFunction(w0,InfiniterHorisontSolver.objectiveFunctionWithD,np.array([1,1]),np.array([1,1]))


#intp=GridLinearInterpolator(idx,w0)

#GraphicsHelpers.plot3D( idx , MathHelpers.EvalFunctionAlongAxis( idx , lambda x,y: intp.FdF(np.array([x,y]))[0] ) , xlabel = "idx 1" , ylabel = "idx 2" , zlabel = "value" )
#GraphicsHelpers.plot3D( idx , MathHelpers.EvalFunctionAlongAxis( idx , lambda x,y: intp.FdF(np.array([x,y]))[1][0] ) , xlabel = "idx 1" , ylabel = "idx 2" , zlabel = "value" )
#GraphicsHelpers.plot3D( idx , MathHelpers.EvalFunctionAlongAxis( idx , lambda x,y: intp.FdF(np.array([x,y]))[1][1] ) , xlabel = "idx 1" , ylabel = "idx 2" , zlabel = "value" )


#w1 = solver.bellmanOperator(w0)
#ret1 = solver.computeFixedPointValueIteration(w0,max_iter = 10)

#GraphicsHelpers.plot3D( idx , ret1.w.T , xlabel = "stock" , ylabel = "price" , zlabel = "value" )
#GraphicsHelpers.plot3D( idx , ret1.sigmas[0].T , xlabel = "stock" , ylabel = "price" , zlabel = "produce" )
#GraphicsHelpers.plot3D( idx , ret1.sigmas[1].T , xlabel = "stock" , ylabel = "price" , zlabel = "store" )

ret2 = solver.computeFixedPointPolicyIteration(w0,max_iter = 10,policy_iter = 100)

GraphicsHelpers.plot3D( idx , ret2.w.T , xlabel = "stock" , ylabel = "price" , zlabel = "value" )
GraphicsHelpers.plot3D( idx , ret2.sigmas[0].T , xlabel = "stock" , ylabel = "price" , zlabel = "produce" )
GraphicsHelpers.plot3D( idx , ret2.sigmas[1].T , xlabel = "stock" , ylabel = "price" , zlabel = "store" )

 
solverKLI = InfiniterHorisontSolver(model, interpolator = KernelLinearInterpolator)

ret3 = solverKLI.computeFixedPointPolicyIteration(w0,max_iter = 10,policy_iter = 100)

GraphicsHelpers.plot3D( idx , ret3.w.T , xlabel = "stock" , ylabel = "price" , zlabel = "value" )
GraphicsHelpers.plot3D( idx , ret3.sigmas[0].T , xlabel = "stock" , ylabel = "price" , zlabel = "produce" )
GraphicsHelpers.plot3D( idx , ret3.sigmas[1].T , xlabel = "stock" , ylabel = "price" , zlabel = "store" )

ret3.sigmas[1]
ret3.sigmas[0]



ret2it = solver.iterateWithFixedPolicy( ret2.w, ret2.sigmas )
ret2.sigmas[1][:,:] = 0
ret2itSigmaFixed = solver.iterateWithFixedPolicy( ret2.w, ret2.sigmas )


GraphicsHelpers.plot3D( idx , (ret2it-ret2itSigmaFixed).T , xlabel = "stock" , ylabel = "price" , zlabel = "value" )
