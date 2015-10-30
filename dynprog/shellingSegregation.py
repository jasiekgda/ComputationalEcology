# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 18:05:39 2015

@author: root
"""

from math import sqrt
import numpy as np
import numpy as np
import matplotlib.pyplot as plt


def distance( x0, y0, x1, y1 ):
    return sqrt( (x0-x1)**2 + (y0-y1)**2 )

class Agent:
    
    def __init__(self, agentType, x = None, y = None):
        self.agentType = agentType
        if x == None or y == None:
            self.x, self.y = np.random.uniform(size = 2)
        else:
            self.x = x
            self.y = y
        
    
    def isHappy( self, agents ):
        agentsWithDistance = map( lambda a: (distance(a.x, a.y, self.x, self.y), 
                                             1 if a.agentType == self.agentType else 0 ), 
                                 agents) 
        agentsWithDistance.sort( key = lambda x: x[0], reverse = True)
        agentsWithDistance = agentsWithDistance[1:11]
        return sum(map(lambda x: x[1], agentsWithDistance) ) >= 5
    
    def moveToNewLocation( self, agents ):
        while not self.isHappy( agents):
            self.x, self.y  = np.random.uniform(size = 2) 
    
    def __repr__(self):
        return "{%s(%s %f %f)" % (self.__class__, self.agentType, self.x , self.y)
    
    def __str__(self):
        return "{%s(%s %f %f)" % (self.__class__, self.agentType, self.x , self.y)

def plotAgents(agents):
    colors = [0 if agent.agentType == 'black' else 1 for agent in agents]
    x = [agent.x for agent in agents]
    y = [agent.y for agent in agents]
    
    plt.scatter(x, y, c=colors)
    plt.show()     
     
agents = [ Agent( "black")  for _ in xrange(250)] + [ Agent( "white")  for _ in xrange(250)]

plotAgents(agents)   

iteration = 0 
while not all([ agent.isHappy( agents ) for agent in agents]):
    print "iter:",iteration,"\n"
    iteration += 1
    for agent in agents:
        agent.moveToNewLocation(agents)

plotAgents(agents)   
        
