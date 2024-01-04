import math
import Utilities as ut
from scipy.sparse.csgraph import minimum_spanning_tree
from tqdm import tqdm, trange
from scipy.sparse import find


def evalSolutions():
    listGreedy = ut.listGreedyTourLength()
    listNearn = ut.listNearnTourLength()
    listFarins = ut.listFarinsTourLength()
    listRandins = ut.listRandinsTourLength()
    listNearins = ut.listNearinsTourLength()
    listCheapins = ut.listCheapinsTourLength()
    listMST = ut.listMSTTourLength()
    listChristo = ut.listChristoTourLength()
    listOptimal = ut.listOptimalTourLength()

    listGreedySolutions = [(((listGreedy[i] - listOptimal[i]) /
                            listOptimal[i]) * 100) for i in range(len(listGreedy))]
    listNearnSolutions = [(((listNearn[i] - listOptimal[i]) /
                           listOptimal[i]) * 100) for i in range(len(listNearn))]
    listFarinsSolutions = [(((listFarins[i] - listOptimal[i]) /
                            listOptimal[i]) * 100) for i in range(len(listFarins))]
    listRandinsSolutions = [(((listRandins[i] - listOptimal[i]) /
                             listOptimal[i]) * 100) for i in range(len(listRandins))]
    listNearinsSolutions = [(((listNearins[i] - listOptimal[i]) /
                             listOptimal[i]) * 100) for i in range(len(listNearins))]
    listCheapinsSolutions = [
        (((listCheapins[i] - listOptimal[i]) / listOptimal[i]) * 100) for i in range(len(listCheapins))]
    listMSTSolutions = [(((listMST[i] - listOptimal[i]) /
                         listOptimal[i]) * 100) for i in range(len(listMST))]
    listChristoSolutions = [(((listChristo[i] - listOptimal[i]) /
                             listOptimal[i]) * 100) for i in range(len(listChristo))]

    return [(listGreedySolutions, round(sum(listGreedySolutions) / len(listGreedySolutions), 2)), (listNearnSolutions, round(sum(listNearnSolutions) / len(listNearnSolutions), 2)), (listFarinsSolutions, round(sum(listFarinsSolutions) / len(listFarinsSolutions), 2)), (listRandinsSolutions, round(sum(listRandinsSolutions) / len(listRandinsSolutions), 2)), (listNearinsSolutions, round(sum(listNearinsSolutions) / len(listNearinsSolutions), 2)), (listCheapinsSolutions, round(sum(listCheapinsSolutions) / len(listCheapinsSolutions), 2)), (listMSTSolutions, round(sum(listMSTSolutions) / len(listMSTSolutions), 2)), (listChristoSolutions, round(sum(listChristoSolutions) / len(listChristoSolutions), 2))]


def amountEqualConnections():
    dimension = ut.listDimension()
    listGreedy = ut.listToTuple(ut.listGreedyTour())
    listNearn = ut.listToTuple(ut.listNearnTour())
    listFarins = ut.listToTuple(ut.listFarinsTour())
    listRandins = ut.listToTuple(ut.listRandinsTour())
    listNearins = ut.listToTuple(ut.listNearinsTour())
    listCheapins = ut.listToTuple(ut.listCheapinsTour())
    listMST = ut.listToTuple(ut.listMSTTour())
    listChristo = ut.listToTuple(ut.listChristoTour())
    listOptimal = ut.listToTuple(ut.listOptimalTour())
    amountGreedy = []
    amountNearn = []
    amountFarins = []
    amountRandins = []
    amountNearins = []
    amountCheapins = []
    amountMST = []
    amountChristo = []
    percentGreedy = []
    percentNearn = []
    percentFarins = []
    percentRandins = []
    percentNearins = []
    percentCheapins = []
    percentMST = []
    percentChristo = []
    for i in trange(len(listGreedy)):
        percentGreedy.append(0)
        percentNearn.append(0)
        percentFarins.append(0)
        percentRandins.append(0)
        percentNearins.append(0)
        percentCheapins.append(0)
        percentMST.append(0)
        percentChristo.append(0)
        amountGreedy.append(0)
        amountNearn.append(0)
        amountFarins.append(0)
        amountRandins.append(0)
        amountNearins.append(0)
        amountCheapins.append(0)
        amountMST.append(0)
        amountChristo.append(0)
        for j in range(len(listGreedy[i])):
            if listGreedy[i][j] in listOptimal[i] or (listGreedy[i][j][1], listGreedy[i][j][0]) in listOptimal[i]:
                amountGreedy[i] = amountGreedy[i] + 1
            if listNearn[i][j] in listOptimal[i] or (listNearn[i][j][1], listNearn[i][j][0]) in listOptimal[i]:
                amountNearn[i] = amountNearn[i] + 1
            if listFarins[i][j] in listOptimal[i] or (listFarins[i][j][1], listFarins[i][j][0]) in listOptimal[i]:
                amountFarins[i] = amountFarins[i] + 1
            if listRandins[i][j] in listOptimal[i] or (listRandins[i][j][1], listRandins[i][j][0]) in listOptimal[i]:
                amountRandins[i] = amountRandins[i] + 1
            if listNearins[i][j] in listOptimal[i] or (listNearins[i][j][1], listNearins[i][j][0]) in listOptimal[i]:
                amountNearins[i] = amountNearins[i] + 1
            if listCheapins[i][j] in listOptimal[i] or (listCheapins[i][j][1], listCheapins[i][j][0]) in listOptimal[i]:
                amountCheapins[i] = amountCheapins[i] + 1
            if listMST[i][j] in listOptimal[i] or (listMST[i][j][1], listMST[i][j][0]) in listOptimal[i]:
                amountMST[i] = amountMST[i] + 1
            if listChristo[i][j] in listOptimal[i] or (listChristo[i][j][1], listChristo[i][j][0]) in listOptimal[i]:
                amountChristo[i] = amountChristo[i] + 1
        percentGreedy[i] = (amountGreedy[i] / dimension[i]) * 100
        percentGreedy[i] = round(percentGreedy[i], 2)
        percentNearn[i] = (amountNearn[i] / dimension[i]) * 100
        percentNearn[i] = round(percentNearn[i], 2)
        percentFarins[i] = (amountFarins[i] / dimension[i]) * 100
        percentFarins[i] = round(percentFarins[i], 2)
        percentRandins[i] = (amountRandins[i] / dimension[i]) * 100
        percentRandins[i] = round(percentRandins[i], 2)
        percentNearins[i] = (amountNearins[i] / dimension[i]) * 100
        percentNearins[i] = round(percentNearins[i], 2)
        percentCheapins[i] = (amountCheapins[i] / dimension[i]) * 100
        percentCheapins[i] = round(percentCheapins[i], 2)
        percentMST[i] = (amountMST[i] / dimension[i]) * 100
        percentMST[i] = round(percentMST[i], 2)
        percentChristo[i] = (amountChristo[i] / dimension[i]) * 100
        percentChristo[i] = round(percentChristo[i], 2)
    return [(percentGreedy, amountGreedy), (percentNearn, amountNearn), (percentFarins, amountFarins), (percentRandins, amountRandins), (percentNearins, amountNearins), (percentCheapins, amountCheapins), (percentMST, amountMST), (percentChristo, amountChristo)]


def mst():
    listNodesV = ut.listNodes()
    listDistances = []
    xd = 0.0
    yd = 0.0
    msts = []
    for i in trange(len(listNodesV)):
        for j in range(len(listNodesV[i])):
            listDistances.append([])
            for k in range(len(listNodesV[i])):
                listDistances[j].append(0)
                xd = listNodesV[i][j][0] - listNodesV[i][k][0]
                yd = listNodesV[i][j][1] - listNodesV[i][k][1]
                listDistances[j][k] = round(math.sqrt(xd * xd + yd * yd))
        msts.append(minimum_spanning_tree(listDistances))
        listDistances = []
    return msts


def evalMSTSolutions():
    msts = mst()
    edges = []
    listOptimal = ut.listOptimalTourLength()
    tour = []
    lengthTour = []
    listMSTSolutions = []
    for i in range(len(msts)):
        row, col, length = find(msts[i])
        edges.append(list(zip(row, col, length)))
        for edge in edges[i]:
            tour.append(edge[2])
        lengthTour.append(sum(tour))
        tour = []
        listMSTSolutions = [((lengthTour[i] / listOptimal[i]) * 100)
                            for i in range(len(lengthTour))]
    return listMSTSolutions


def mstEqualConnections():
    dimension = ut.listDimension()
    msts = mst()
    listOptimal = ut.listToTuple(ut.listOptimalTour())
    amountMST = []
    percentMST = []
    edges = []
    for i in range(len(msts)):
        row, col, _ = find(msts[i])
        edges.append(list(zip(col, row)))
    for i in trange(len(edges)):
        amountMST.append(0)
        percentMST.append(0)
        for j in range(len(edges[i])):
            if edges[i][j] in listOptimal[i] or (edges[i][j][1], edges[i][j][0]) in listOptimal[i]:
                amountMST[i] = amountMST[i] + 1
        percentMST[i] = (amountMST[i] / dimension[i]) * 100
        percentMST[i] = round(percentMST[i], 2)
    return (percentMST, amountMST)


def kRelativeNeighborhoodGraph(k=1):
    nodes = ut.listNodes()
    nodedic = {}
    amount = 0
    graph = []
    for i in range(len(nodes)):
        nodedic[i] = {}
        for j in range(len(nodes[i])):
            nodedic[i][j] = (nodes[i][j][0], nodes[i][j][1])
    for i in trange(len(nodes)):
        graph.append([])
        for j in range(len(nodes[i])):
            for l in range(len(nodes[i])):
                if nodes[i][j] == nodes[i][l]:
                    continue
                distance_n1_n2 = distance(nodes[i][j], nodes[i][l])
                for m in range(len(nodes[i])):
                    if nodes[i][m] != nodes[i][j] and nodes[i][m] != nodes[i][l]:
                        distance_n3_n1 = distance(
                            nodes[i][m], nodes[i][j])
                        distance_n3_n2 = distance(
                            nodes[i][m], nodes[i][l])
                        if distance_n3_n1 < distance_n1_n2 and distance_n3_n2 < distance_n1_n2:
                            amount = amount + 1
                            if amount >= k:
                                break
                if amount < k:
                    if (l, j) not in graph:
                        graph[i].append((j, l))
                amount = 0
    return (graph, nodedic)


def smallestKInOptimal():
    listOptimal = ut.listToTuple(ut.listOptimalTour())
    listOptimalNodes = ut.listOptimalNodes()
    result = [['k', 'absoluten Zahlen', 'Prozentanteile']]
    k = []
    amount = 0
    for i in trange(len(listOptimal)):
        for edge in listOptimal[i]:
            for j in range(len(listOptimalNodes[i])):
                if j != edge[0] and j != edge[1]:
                    distance_v_i = distance(
                        listOptimalNodes[i][j], listOptimalNodes[i][edge[0]])
                    distance_i_j = distance(
                        listOptimalNodes[i][edge[0]], listOptimalNodes[i][edge[1]])
                    distance_v_j = distance(
                        listOptimalNodes[i][j], listOptimalNodes[i][edge[1]])
                    if distance_v_i < distance_i_j and distance_v_j < distance_i_j:
                        amount = amount + 1
            k.append(amount)
            amount = 0
    for i in range(max(k)):
        amount = k.count(i)
        percent = round((amount / len(k)) * 100, 2)
        result.append([i, amount, percent])
    return (k, result)


def distance(node1, node2):
    x1 = node1[0]
    x2 = node2[0]
    y1 = node1[1]
    y2 = node2[1]
    x12 = x1 - x2
    y12 = y1 - y2
    return round(math.sqrt(x12**2 + y12**2))


def optimalInRNG(rng):
    dimension = ut.listDimension()
    listOptimal = ut.listToTuple(ut.listOptimalTour())
    amount = 0
    percent = []

    for i in range(len(rng)):
        for j in range(len(listOptimal[i])):
            if listOptimal[i][j] in rng[i]:
                amount = amount + 1
        percent.append(amount / dimension[i])
        amount = 0
    return percent


def greedyInRNG(rng):
    dimension = ut.listDimension()
    listGreedy = ut.listToTuple(ut.listGreedyTour())
    amount = 0
    percent = []

    for i in range(len(rng)):
        for j in range(len(listGreedy[i])):
            if listGreedy[i][j] in rng[i]:
                amount = amount + 1
        percent.append(amount / dimension[i])
        amount = 0
    return percent
