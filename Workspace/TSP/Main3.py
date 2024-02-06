import Utilities as ut
import math
import json
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import matplotlib.pyplot as plt

# Distanzfunktion zwischen Punkten
# Theoretisch kann es sein, dass die Distanz zwischen zwei Punkten
# gleich 0 ist. Daher + 1 für die Distanz, da dies keinen Einfluss auf das
# Minimalgerüst hat.
def distanz(p1, p2):
    xdelta = p1['x'] - p2['x']
    ydelta = p1['y'] - p2['y']
    return int(round(math.sqrt(xdelta * xdelta + ydelta * ydelta))) + 1

nodes = ut.listNodes()[0]
n = len(nodes)				                            # Problemgröße			              
points = [{'x' : p[0], 'y': p[1]} for p in nodes]	# Koordinaten umwandeln

matrix = [[0 for i in range(n)] for j in range(n)]	# Distanzmatrix für
for i in range(n):					# MST-Funktion bauen
    for j in range(i+1,n):
        matrix[i][j] = distanz(points[i], points[j])
        matrix[j][i] = matrix[i][j]

graph = csr_matrix(matrix)				# MST berechnen
tcsr = minimum_spanning_tree(graph)
edges = np.array(tcsr.nonzero()).T
print(len(edges))
print(tcsr)
tree = tcsr.toarray().astype(int)			# Kanten des MST bestimmen
edgelist = []
for i in range(n):
    for j in range(n):
        if tree[i][j] != 0: 
            edgelist.append([i,j])

for e in edgelist:					# MST plotten
    x1 = points[e[0]]['x']
    x2 = points[e[1]]['x']
    y1 = points[e[0]]['y']
    y2 = points[e[1]]['y']
    plt.plot([x1,x2],[y1,y2],'bo', linestyle='-')
plt.show()

for e in edgelist:				# Jetzt alle Kanten des MST aus
    i = e[0]					# dem Graph entfernen. Hierzu
    j = e[1]					# wird der Distanzwert in der Matrix
    matrix[i][j] = 0				# auf 0 gesetzt, so dass der Algorithmus
    matrix[j][i] = 0				# diese Kante nicht mehr berücksichtigt.

graph = csr_matrix(matrix)			# MST auf Graph ohne Kanten des ersten
tcsr = minimum_spanning_tree(graph)		# MST berechnen
edges = np.array(tcsr.nonzero()).T
print(len(edges))
print(tcsr)
tree = tcsr.toarray().astype(int)		# jetzt wieder die Kanten bestimmen und
edgelist = []
for i in range(n):
    for j in range(n):
        if tree[i][j] != 0: 
            edgelist.append([i,j])

for e in edgelist:				# den zweiten MST plotten
    x1 = points[e[0]]['x']
    x2 = points[e[1]]['x']
    y1 = points[e[0]]['y']
    y2 = points[e[1]]['y']
    plt.plot([x1,x2],[y1,y2],'bo', linestyle='-')
plt.show()