# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 18:35:42 2015

@author: root
"""
 
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter


class MathHelpers:
    @staticmethod
    def GaussNoise( n , mu, sigma ):
        x, w = np.polynomial.hermite.hermgauss(n)
        return [math.sqrt(2)*sigma*x+mu, w/math.sqrt(math.pi)]
        
    @staticmethod
    def EvalFunctionAlongAxis( idxs, fn ):        
        if len(idxs) > 2:
            raise Exception('only 2d array supported')
        X, Y = np.meshgrid(idxs[0], idxs[1])
        return [map( lambda crd: fn(*crd) , zip(x,y)) for x,y in zip(X,Y)]


class GraphicsHelpers:
    
    @staticmethod
    def plotHistory( states, events, actions, stateLabels, actionLabels ):
        shocks = np.where(events == 0 )[0]
        ticks = [0]*shocks.shape[0]

        fig, ax = plt.subplots()
        x = np.linspace(0, states.shape[0] , states.shape[0])
        for i in range(states.shape[1]):
    
            y = states[:,i]
            current_label = r'state {0}'.format(stateLabels[i])
            ax.plot(x, y, linewidth=2, alpha=0.6, label=current_label)
    
        ax.plot(shocks, ticks, 'ro')    
    
        ax.legend()
        plt.show()


        for i in range(states.shape[1]):

            fig, ax = plt.subplots()
            x = np.linspace(0, states.shape[0] , states.shape[0])
    
            y = states[:,i]
            current_label = r'state {0}'.format(stateLabels[i])
            ax.plot(x, y, linewidth=2, alpha=0.6, label=current_label)
    
            ax.plot(shocks, ticks, 'ro')    
            ax.legend()
            plt.show()

        for i in range(actions.shape[1]):

            fig, ax = plt.subplots()
            x = np.linspace(0, actions.shape[0] , actions.shape[0])
    
            y = actions[:,i]
            current_label = r'action {0}'.format(actionLabels[i])
            ax.plot(x, y, linewidth=2, alpha=0.6, label=current_label)
    
            ax.plot(shocks, ticks, 'ro')    
            ax.legend()
            plt.show()
    
    @staticmethod
    def plot3D( idx , Z, xlabel = "X", ylabel = "Y", zlabel = "Z", ticks = 5, fmt = '%.02f'):
        xmin = np.min(idx[0])
        ymax = np.max(idx[1])
        zmin = np.min(Z)
        zmax = np.max(Z)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        X, Y = np.meshgrid(idx[0], idx[1])
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False, alpha=0.3)
        
        cset = ax.contour(X, Y, Z, zdir='z', offset=zmin, cmap=cm.coolwarm)
        cset = ax.contour(X, Y, Z, zdir='x', offset=xmin, cmap=cm.coolwarm)
        cset = ax.contour(X, Y, Z, zdir='y', offset=ymax, cmap=cm.coolwarm)
         
        ax.set_zlim(zmin, zmax)
        ax.zaxis.set_major_locator(LinearLocator(ticks))
        ax.zaxis.set_major_formatter(FormatStrFormatter(fmt))

        ax.xaxis.set_major_locator(LinearLocator(ticks))
        ax.xaxis.set_major_formatter(FormatStrFormatter(fmt))
        ax.yaxis.set_major_locator(LinearLocator(ticks))
        ax.yaxis.set_major_formatter(FormatStrFormatter(fmt))
        
        ax.set_xlabel( xlabel )
        ax.set_ylabel( ylabel )
        ax.set_zlabel( zlabel )

        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()

class Struct:
    def __init__(self, **entries): self.__dict__.update(entries)