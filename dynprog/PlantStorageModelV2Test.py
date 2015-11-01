# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 12:51:28 2015

@author: root
"""

import pickle
from DynamicProgramming.DynamicsSymulation import *
from DynamicProgramming.Interpolators import *
from DynamicProgramming.Helpers import *
from DynamicProgramming.DynamicModel import *
from DynamicProgramming.InfiniteHorisont import *
from DynamicProgramming.PlantStorageModel import *



kappa = 0.75
def P(w):
    return 0.1*w**kappa    

plantStorageModel = PlantStorageModelV1( P = P, beta = 0.95 )
idx = plantStorageModel.getIdx( grid_min = [0.0,0.0 ], 
                                #grid_max = [4.0,4.0], 
                                #grid_size = [6,6]
                                grid_max = [1.0,1.0], 
                                grid_size = [6,6]
                                ) #[11,11]

print "model version: ",plantStorageModel.ver()

w0 = plantStorageModel.init(idx)
model = DynamicModelWithGradOptimisation(
                     plantStorageModel.utility,
                     plantStorageModel.dUtility,
                     plantStorageModel.g,
                     plantStorageModel.dG,
                     plantStorageModel.la,
                     plantStorageModel.ua,
                     idx,
                     plantStorageModel.beta,
                     [[0,0.05],[1,0.95]])

print (-np.log(0.98*0.95)/(0.1*0.75))**-4

solverWithDiff = InfiniterHorisontSolver(model, interpolator = GridLinearInterpolator)
retDiff = solverWithDiff.computeFixedPointPolicyIteration(w0,error_tol = 1e-2, max_iter =50,max_policy_iter = 500)

DynamicModelWithGradOptimisation.printErrors(solverWithDiff.optimisationHistory)


retDiffFullCutHiBeta = retDiff
#retDiffPartCut = retDiff
#retDiffFullCut = retDiff

idxRev = [idx[1],idx[0]]


GraphicsHelpers.plot3D( idxRev , w0 , xlabel = "biomas" , ylabel = "storage" , zlabel = "value" )
GraphicsHelpers.plot3D( idxRev , retDiff.w , xlabel = "biomas" , ylabel = "storage" , zlabel = "value" )
GraphicsHelpers.plot3D( idxRev , retDiff.sigmas[0] , xlabel = "biomas" , ylabel = "storage" , zlabel = "a0 - transfer from storage" )
GraphicsHelpers.plot3D( idxRev , retDiff.sigmas[1] , xlabel = "biomas" , ylabel = "storage" , zlabel = "a1 - transfer to growth" )

simulator = DynamicsSymulation( model, retDiff.sigmas)
states, events, actions = simulator.simulateTrial( 500 , np.array([0,0.0001]))



GraphicsHelpers.plotHistory(states, events, actions,
                            stateLabels = ["storage","size"],
                            actionLabels = ["p1","p2"]
                            )

np.where(events==0)
states[300:400,0]

#import matplotlib.pyplot as plt
#import numpy as np
#fig, ax = plt.subplots()
#x = np.linspace(0, 2, 201)
#y = P(x)
#ax.plot(x, y, 'b-', linewidth=2)
#plt.show()


with open('retDiffFullCutHiBeta_data.pkl', 'wb') as output:
    pickle.dump(retDiffFullCutHiBeta, output, pickle.HIGHEST_PROTOCOL)
#((0.1,0.05),(1,0.95))
#beta = 0.99

with open('retDiffPartCut_data.pkl', 'wb') as output:
    pickle.dump(retDiffPartCut, output, pickle.HIGHEST_PROTOCOL)
#((0.1,0.05),(1,0.95))

with open('retDiffFullCut_data.pkl', 'wb') as output:
    pickle.dump(retDiffFullCut, output, pickle.HIGHEST_PROTOCOL)
#((0,0.05),(1,0.95))



#with open('company_data.pkl', 'rb') as input:
#    company1 = pickle.load(input)
#    print(company1.name)  # -> banana
#    print(company1.value)  # -> 40