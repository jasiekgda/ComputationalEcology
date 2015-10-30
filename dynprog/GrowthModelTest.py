# -*- coding: utf-8 -*-
from DynamicProgramming.InfiniteHorisont import *
from DynamicProgramming.GrowthModel import *
import matplotlib.pyplot as plt

idx = GrowthModel.getIdx()
w0 = GrowthModel.init(idx)
model = DynamicModel(GrowthModel.utility,
                     GrowthModel.g,
                     GrowthModel.la,
                     GrowthModel.ua,
                     idx,
                     GrowthModel.beta)



solver = ValueIterationSolver(model)
ret = solver.computeFixedPoint(w0,max_iter = 5)
#model.g(np.array([0.1]),np.array([0.2]),0)
#solver.bellmanOperator(w0)


grid = idx[0]

fig, ax = plt.subplots()
ax.set_ylim(-40, -20)
ax.set_xlim(np.min(grid), np.max(grid))
lb = 'initial condition'
ax.plot(grid, w0, color=plt.cm.jet(0), lw=2, alpha=0.6, label=lb)
lb = 'results'
ax.plot(grid, ret.w, 'r-', lw=2, alpha=0.6, label = lb)
lb = 'true value function'
ax.plot(grid, GrowthModel.vStar(grid), 'k-', lw=2, alpha=0.8, label=lb)
ax.legend(loc='upper left')
plt.show()

fig, ax = plt.subplots()
ax.set_ylim(np.min(ret.sigmas[0]), np.max(ret.sigmas[0]))
ax.set_xlim(np.min(grid), np.max(grid))
lb = 'policy'
ax.plot(grid, ret.sigmas[0], 'k-', lw=2, alpha=0.6, label=lb)
ax.legend(loc='upper left')
plt.show()
