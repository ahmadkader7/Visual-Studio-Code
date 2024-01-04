import Features as feat
import matplotlib.pyplot as plt
import numpy as np
import tabulate as tab
import Utilities as ut
import Calculations as cal
import Plots as plo
import networkx as nx

listTourLength = cal.evalSolutions()
listTour = cal.amountEqualConnections()
listMSTTour = cal.mstEqualConnections()
listMSTTourLength = cal.evalMSTSolutions()

tupleGreedy = listTourLength[0][0]
tupleGreedy2 = listTour[0][0]
tupleNearn = listTourLength[1][0]
tupleNearn2 = listTour[1][0]
tupleFarins = listTourLength[2][0]
tupleFarins2 = listTour[2][0]
tupleRandins = listTourLength[3][0]
tupleRandins2 = listTour[3][0]
tupleNearins = listTourLength[4][0]
tupleNearins2 = listTour[4][0]
tupleCheapins = listTourLength[5][0]
tupleCheapins2 = listTour[5][0]
tupleMST = listTourLength[6][0]
tupleMST2 = listTour[6][0]
tupleChristo = listTourLength[7][0]
tupleChristo2 = listTour[7][0]
tupleMSTS = listMSTTour[1]
tupleMSTS2 = listMSTTour[0]

SolutionAverages = [
    listTourLength[0][1],
    listTourLength[1][1],
    listTourLength[2][1],
    listTourLength[3][1],
    listTourLength[4][1],
    listTourLength[5][1],
    listTourLength[6][1],
    listTourLength[7][1],
    (sum(listMSTTourLength) / len(listMSTTourLength)),
]
percentages = [
    tupleGreedy2,
    tupleNearn2,
    tupleFarins2,
    tupleRandins2,
    tupleNearins2,
    tupleCheapins2,
    tupleMST2,
    tupleChristo2,
    tupleMSTS2,
]
percentagesAverages = [
    round(sum(percentages[i]) / len(percentages[i]), 2) for i in range(len(percentages))
]

plo.plotter(tupleGreedy, tupleGreedy2, "greedy")
plo.plotter(tupleNearn, tupleNearn2, "nearest neighbor")
plo.plotter(tupleFarins, tupleFarins2, "farthest insertion")
plo.plotter(tupleRandins, tupleRandins2, "random insertion")
plo.plotter(tupleNearins, tupleNearins2, "nearest insertion")
plo.plotter(tupleCheapins, tupleCheapins2, "cheapest insertion")
plo.plotter(tupleMST, tupleMST2, "mst heuristic")
plo.plotter(tupleChristo, tupleChristo2, "christofides")
plo.plotter(listMSTTourLength, tupleMSTS2, "minimum spanning tree")
plo.scatterPlotter(SolutionAverages, percentagesAverages)

res = cal.kRelativeNeighborhoodGraph(k=1)

G1 = nx.Graph(res[0][0])

nx.draw(G1, pos=res[1][0], node_color="r", node_size=20, edge_color="b", width=0.5)

plt.show()

rng = cal.kRelativeNeighborhoodGraph(k=1)[0]

sol1 = cal.optimalInRNG(rng)
sol2 = cal.greedyInRNG(rng)

print(sol1)
print(sol2)

plo.plotter(sol1, sol2, "1-RNG")

k = cal.smallestKInOptimal()[0]
plt.hist(k, bins=np.arange(8) - 0.5, edgecolor="black")
plt.xlim([-0.5, 8.5])
plt.title("Smallest k in edge from optimal solutions")
plt.xlabel("k")
plt.ylabel("amount of instances")
plt.show()

result = cal.smallestKInOptimal()[1]
print(tab.tabulate(result, headers="firstrow", tablefmt="fancy_grid"))


v = ut.listNodes()[20]
mst = feat.mst(v)
points = np.array(v)

edges = np.array(mst.nonzero()).T

# Plot the original points
plt.scatter(points[:, 0], points[:, 1], c="b", marker="o", label="Points")

# Plot the edges of the MST
for edge in edges:
    plt.plot(points[edge, 0], points[edge, 1], "r-")

plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Minimum Spanning Tree")
plt.legend(["MST Edges", "Points"])
plt.grid(True)
plt.show()

instance = 350
v = ut.listOptimalNodes()[instance]
e_all = ut.listToTuple(ut.listGreedyTour())
e = e_all[instance]
result = feat.mstFeature(v, e)
y = []
for i in range(len(result)):
    y.append(result[i][1])
print(y)
bin_edges = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
bin_width = 1.0

plt.hist(y, bins=bin_edges, edgecolor="black", rwidth=bin_width)
plt.xticks(range(10))
plt.yticks(range(0, 250, 5))
plt.xlabel("k")
plt.ylabel("Anzahl Kanten")
plt.title("MST Merkmal")
plt.grid(axis="y", linestyle="--", alpha=0.7)

plt.show()

instance = 350
v = ut.listNodes()[instance]
e_all = ut.listToTuple(ut.listGreedyTour())
e = e_all[instance]
nodedic = {}
for i in range(len(v)):
    nodedic[i] = (v[i][0], v[i][1])
# result = feat.mstFeature(v, e)
mst = feat.mst(v, e)
mstold = feat.mstOld(v)

points = np.array(v)

edges = np.array(mstold.nonzero()).T

# Plot the original points
plt.scatter(points[:, 0], points[:, 1], c="b", marker="o", label="Points")

# Plot the edges of the MST
for edge in edges:
    plt.plot(points[edge, 0], points[edge, 1], "r-")

plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Minimum Spanning Tree")
plt.legend(["MST Edges", "Points"])
plt.grid(True)
plt.show()

instance = 0
v = ut.listNodes()[instance]
result = feat.mstFeature(v)

points = np.array(v)
mst = 1
edges = np.array(result[1][mst].nonzero()).T

# Plot the original points
plt.scatter(points[:, 0], points[:, 1], c="b", marker="o", label="Points")

# Plot the edges of the MST
for edge in edges:
    plt.plot(points[edge, 0], points[edge, 1], "r-")

plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Minimum Spanning Tree")
plt.legend(["MST Edges", "Points"])
plt.grid(True)
plt.show()

instance = 0
v = ut.listNodes()[instance]
result = feat.mstFeature(v)
