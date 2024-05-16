import math
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix
from scipy.spatial import distance_matrix
from tqdm import tqdm, trange

def mstFeature(vertices, greedy_edges, k=10):
    mst_edges = []
    values = []
    
    for m in trange(len(vertices)):
        mst_edges_instance = []
        values_unsorted = []
        greedy_edges_instance = greedy_edges[m]
        points = [{'x' : p[0], 'y': p[1]} for p in vertices[m]]
        n = len(points)
        matrix = createMatrix(points, n)
        graph = csr_matrix(matrix)
        for i in range(k):
            resultMST = minimum_spanning_tree(graph)
            tree = resultMST.toarray().astype(int)
            edgelist = []
            for j in range(n):
                for l in range(n):
                    if tree[j][l] != 0:
                        edgelist.append([j, l])
                        if (j, l) in greedy_edges_instance or (l, j) in greedy_edges_instance:
                            mst_edges_instance.append((j, l))
                            values_unsorted.append(i + 1)
            matrix = updateMatrix(matrix, edgelist)
            graph = csr_matrix(matrix)
        for j in range(len(greedy_edges_instance)):
            x = greedy_edges_instance[j][0]
            y = greedy_edges_instance[j][1]
            if (x, y) not in mst_edges_instance and (y, x) not in mst_edges_instance:
                mst_edges_instance.append((x, y))
                values_unsorted.append(2 * k)
        mst_edges_instance, values_unsorted = sort_values_to_greedy(mst_edges_instance, greedy_edges_instance, values_unsorted)
        mst_edges.extend(mst_edges_instance)
        values.extend(values_unsorted)
    return values

def createMatrix(points, n):
    matrix = [[0 for i in range(n)] for j in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j] = distance(points[i], points[j])
            matrix[j][i] = matrix[i][j]
    return matrix


def updateMatrix(matrix, edgelist):
    updatedMatrix = matrix.copy()
    for e in edgelist:
        i = e[0]
        j = e[1]
        updatedMatrix[i][j] = 0
        updatedMatrix[j][i] = 0
    return updatedMatrix

def localFeatures(vertices, edges):
    f1, f2, f3, f4, f5, f6 = []
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
    f1, f2, f3, f4 = []
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
            if (x, y) in cheapins[instance_number] or (y, x) in cheapins[instance_number]:
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
            v1 = edge[0]
            v2 = edge[1]
            if (v1, v2) in edges_optimal[i] or (v2, v1) in edges_optimal[i]:
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

def sort_values_to_greedy(mst_edges, greedy_edges_instance, values):
    sorted_edges_set = set(greedy_edges_instance)
    needs_reversal = set()
    
    for edge in mst_edges:
        if edge not in sorted_edges_set and (edge[1], edge[0]) in sorted_edges_set:
            needs_reversal.add(edge)

    adjusted_unsorted_edges = [(e[1], e[0]) if e in needs_reversal else e for e in mst_edges]

    edge_to_sorted_index = {edge: i for i, edge in enumerate(greedy_edges_instance)}

    indexed_values = [(edge_to_sorted_index[edge], value) for edge, value in zip(adjusted_unsorted_edges, values)]
    indexed_values.sort(key=lambda x: x[0])

    sorted_values = [value for _, value in indexed_values]

    sorted_unsorted_edges = sorted(adjusted_unsorted_edges, key=lambda edge: edge_to_sorted_index[edge])

    assert sorted_unsorted_edges == greedy_edges_instance, "Edges not sorted or adjusted correctly"
    return sorted_unsorted_edges, sorted_values