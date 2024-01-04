import math
import Utilities as ut
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix
from scipy.spatial import distance_matrix
from tqdm import tqdm, trange
from scipy.sparse import find


def mstFeature(vertices, k=10):
    result = []
    n = len(vertices)
    points = [{"x": p[0], "y": p[1]} for p in vertices]
    matrix = createMatrix(points, n)
    for i in range(k):
        print(matrix)
        resultMST = minimum_spanning_tree(matrix)
        tree = resultMST.toarray().astype(int)
        edgelist = []
        for j in range(n):
            for l in range(n):
                if tree[j][l] != 0:
                    edgelist.append([j, l])
                    result.append(((j, l), (i + 1)))
        matrix = updateMatrix(matrix, edgelist)
    return result


def createMatrix(points, n):
    matrix = [[0 for i in range(n)] for j in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j] = distance(points[i], points[j])
            matrix[j][i] = matrix[i][j]
    return csr_matrix(matrix)


def updateMatrix(matrix, edgelist):
    updatedMatrix = matrix.copy()
    for e in edgelist:
        i = e[0]
        j = e[1]
        updatedMatrix[i, j] = 0
        updatedMatrix[j, i] = 0
    return updatedMatrix


def localFeatures(vertices, edges):
    f1 = []
    f2 = []
    f3 = []
    f4 = []
    f5 = []
    f6 = []
    for k in trange(len(vertices), desc="localfeatures"):
        points = np.array(vertices[k])
        distances = distance_matrix(points, points)
        max_weight = np.max(distances)
        min_weight = minimum_distance(distances)
        for edge in edges[k]:
            i = edge[0]
            j = edge[1]
            max_weight_i = np.max(distances[i, :])
            max_weight_j = np.max(distances[:, j])
            min_weight_i = np.min(np.delete(distances[i, :], i))
            min_weight_j = np.min(np.delete(distances[:, j], j))

            f1.append((1 + distances[i][j]) / (1 + max_weight))
            f2.append((1 + distances[i][j]) / (1 + max_weight_i))
            f3.append((1 + distances[i][j]) / (1 + max_weight_j))
            f4.append((1 + min_weight) / (1 + distances[i][j]))
            f5.append((1 + min_weight_i) / (1 + distances[i][j]))
            f6.append((1 + min_weight_j) / (1 + distances[i][j]))

    return (f1, f2, f3, f4, f5, f6)


def graphFeatures(vertices, edges):
    f1 = []
    f2 = []
    f3 = []
    f4 = []
    for k in trange(len(vertices), desc="graphfeatures"):
        points = np.array(vertices[k])
        distances = distance_matrix(points, points)
        n = len(distances[0])
        for edge in edges[k]:
            i = edge[0]
            j = edge[1]
            max_weight_i = np.max(distances[i, :])
            max_weight_j = np.max(distances[:, j])
            min_weight_i = np.min(np.delete(distances[i, :], i))
            min_weight_j = np.min(np.delete(distances[:, j], j))
            sum_weight_i = np.sum(distances[i, :])
            sum_weight_j = np.sum(distances[:, j])
            f1.append((distances[i][j] - min_weight_i) / (max_weight_i - min_weight_i))
            f2.append((distances[i][j] - min_weight_j) / (max_weight_j - min_weight_j))
            f3.append(
                (distances[i][j] - (sum_weight_i / n)) / (max_weight_i - min_weight_i)
            )
            f4.append(
                (distances[i][j] - (sum_weight_j / n)) / (max_weight_j - min_weight_j)
            )

    return (f1, f2, f3, f4)


def heuristicsFeature(greedy, nearn, fartins, randins, nearins, cheapins, mst, christo):
    result_nearn = []
    result_fartins = []
    result_randins = []
    result_nearins = []
    result_cheapins = []
    result_mst = []
    result_christo = []
    instance_number = -1
    for instance in tqdm(greedy, desc="heuristicsfeature"):
        instance_number = instance_number + 1
        for edge in instance:
            x = edge[0]
            y = edge[1]
            if (x, y) in nearn[instance_number] or (y, x) in nearn[instance_number]:
                result_nearn.append(1)
            else:
                result_nearn.append(0)
            if (x, y) in fartins[instance_number] or (y, x) in fartins[instance_number]:
                result_fartins.append(1)
            else:
                result_fartins.append(0)
            if (x, y) in randins[instance_number] or (y, x) in randins[instance_number]:
                result_randins.append(1)
            else:
                result_randins.append(0)
            if (x, y) in nearins[instance_number] or (y, x) in nearins[instance_number]:
                result_nearins.append(1)
            else:
                result_nearins.append(0)
            if (x, y) in cheapins[instance_number] or (y, x) in cheapins[
                instance_number
            ]:
                result_cheapins.append(1)
            else:
                result_cheapins.append(0)
            if (x, y) in mst[instance_number] or (y, x) in mst[instance_number]:
                result_mst.append(1)
            else:
                result_mst.append(0)
            if (x, y) in christo[instance_number] or (y, x) in christo[instance_number]:
                result_christo.append(1)
            else:
                result_christo.append(0)

    return (
        result_nearn,
        result_fartins,
        result_randins,
        result_nearins,
        result_cheapins,
        result_mst,
        result_christo,
    )


def kRNGFeature(vertices, edges):
    k = []
    amount = 0
    for i in trange(len(edges), desc="krngfeature"):
        for edge in edges[i]:
            for j in range(len(vertices[i])):
                if j != edge[0] and j != edge[1]:
                    distance_v_i = distance2(vertices[i][j], vertices[i][edge[0]])
                    distance_i_j = distance2(vertices[i][edge[0]], vertices[i][edge[1]])
                    distance_v_j = distance2(vertices[i][j], vertices[i][edge[1]])
                    if distance_v_i < distance_i_j and distance_v_j < distance_i_j:
                        amount = amount + 1
            k.append(amount)
            amount = 0
    return k


def sumHeuristicsFeature(nearn, fartins, randins, nearins, cheapins, mst, christo):
    sums = []
    for i in trange(len(nearn), desc="sumHeuristicsfeature"):
        sums.append(
            nearn[i]
            + fartins[i]
            + randins[i]
            + nearins[i]
            + cheapins[i]
            + mst[i]
            + christo[i]
        )
    return sums


def calculateTargets(edges_optimal, edges_greedy):
    targets = []
    for i in range(len(edges_greedy)):
        for edge in edges_greedy[i]:
            if edge in edges_optimal[i]:
                targets.append(1)
            else:
                targets.append(0)
    return targets


def minimum_distance(distances):
    min = np.max(distances)
    for i in range(len(distances)):
        for j in range(len(distances[i])):
            d = distances[i][j]
            if i != j and d < min:
                min = d
    return min


def distance(p1, p2):
    xdelta = p1["x"] - p2["x"]
    ydelta = p1["y"] - p2["y"]
    return int(round(math.sqrt(xdelta * xdelta + ydelta * ydelta))) + 1


def distance2(node1, node2):
    x1 = node1[0]
    x2 = node2[0]
    y1 = node1[1]
    y2 = node2[1]
    x12 = x1 - x2
    y12 = y1 - y2
    return round(math.sqrt(x12**2 + y12**2))
