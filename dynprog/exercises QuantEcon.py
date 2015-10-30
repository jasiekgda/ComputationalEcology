# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 19:21:53 2015

@author: root
"""

import numpy as np

dataFile = open( 'C:\\Anaconda - skrypty\\dynprog\\graph1.txt' )
lines = [line.split(',') for line in dataFile]
dataFile.close()
connections = map( lambda x : (x[0], filter( lambda x: x!= [''] , [ngh.strip(' \t\n\r').split(' ') for ngh in x[1:]])),lines)
nodesCnt = len(connections)

edges = np.ndarray( [ nodesCnt, nodesCnt] , dtype = float )
edges[:,:] = float('inf')

nameToIdDict = dict()
idToNameDict = dict()
pos = 0

for vertex in connections:
    nameToIdDict[vertex[0]]=pos
    idToNameDict[pos]=vertex[0]
    pos += 1

for vertex in connections:
    edges[nameToIdDict[vertex[0]],nameToIdDict[vertex[0]]] = 0
    for ngh in vertex[1]:
        edges[nameToIdDict[vertex[0]],nameToIdDict[ngh[0]]]=ngh[1]
        


#dijkstra shortest path



import heapq

def dijkstra(G, s):
    nodesCnt = G.shape[0]

    d = np.ndarray( [nodesCnt], dtype = float )
    d[:] = float('inf')
    d[s] = 0

    Q = []

    for node, val in enumerate(d):
        heapq.heappush( Q , ( val, node ))

    while len(Q) > 0:
        _, u = heapq.heappop(Q)
        for v, _ in filter( lambda x: 0 < x[1] < float('inf'),enumerate(G[u,:])):
            print v," ",u
            if d[v] > d[u] + G[u,v]:
                d[v] = d[u] + G[u,v]
                heapq.heappush( Q, (d[v], v))
    
    return d

G = edges
s = 0    
dijkstra(edges,0)

#dynamic programing shortest path

Jprev = np.ndarray([nodesCnt],dtype = float)
Jprev[:] = float('inf')
Jprev[nameToIdDict['node99']]= 0

Js=[]
while True:
    Jcurrent = np.ndarray([nodesCnt],dtype = float)    
    Acurrent = np.ndarray([nodesCnt],dtype = int)
    
    for v in idToNameDict.iterkeys():
        val, act = min( [ ( Jprev[w] + edges[v,w], w ) for w in idToNameDict.iterkeys()])        
        Jcurrent[v], Acurrent[v] = val, act
                
    
    if (Jcurrent == Jprev).all():
        break
    
    
    Js.append([Jcurrent,Acurrent])
    Jprev = Jcurrent

len(Js)
Js[24][0][0]



path = []
path.append( Js[len(Js)-1][1][0] )

for i in xrange(len(Js)-2,-1,-1):
    lastIdx = path.pop()
    path.append(lastIdx)
    path.append(Js[i][1][lastIdx])
    

node0
node8
node11
node18
node23
node33
node41
node53
node56
node57
node60
node67
node70
node73
node76
node85
node87
node88
node93
node94
node96
node97
node98
node99

Cost:  160.55