import json

def readJSON(datadir):
    f = open(datadir)
    data = json.load(f)
    return data


def readSpecialCords(datadir):
    f = open(datadir)
    data = json.load(f)
    return data["node_coordinates"]


def readSpecialEdges(datadir):
    f = open(datadir)
    data = json.load(f)
    return data["edge_data"]


def readTour(datadir):
    f = open(datadir)
    data = json.load(f)
    return data["tour"]


def readTourLength(datadir):
    f = open(datadir)
    data = json.load(f)
    return data["tourlength"]


def readDimension(datadir):
    f = open(datadir)
    data = json.load(f)
    return data["dimension"]


def readNodes(datadir):
    f = open(datadir)
    data = json.load(f)
    return data["node_coordinates"]


def printJSON(datadir):
    f = open(datadir)
    data = json.load(f)
    print(data)


def listSpecialCords():
    return readSpecialCords("Workspace/TSP/Data/tsp_500_s816.json")


def listSpecialEdges():
    return readSpecialEdges("Workspace/TSP/Data/tsp_500_s816.json")


def listDimension():
    return [
        readDimension("Workspace/TSP/Data/fullRandom_large_" + str(x) + ".json")
        for x in range(1000)
    ]


def listNodes():
    return [
        readNodes("Workspace/TSP/Data/fullRandom_large_" + str(x) + ".json")
        for x in range(1000)
    ]


def listOptimalNodes():
    return [
        readNodes("Workspace/TSP/Data/fullRandom_large_" + str(x) + "_sol.json")
        for x in range(1000)
    ]


def listGreedyTourLength():
    return [
        readTourLength("Workspace/TSP/Data/fullRandom_large_" + str(x) + "_greedy.json")
        for x in range(1000)
    ]


def listNearnTourLength():
    return [
        readTourLength("Workspace/TSP/Data/fullRandom_large_" + str(x) + "_nearn.json")
        for x in range(1000)
    ]


def listFarinsTourLength():
    return [
        readTourLength("Workspace/TSP/Data/fullRandom_large_" + str(x) + "_farins.json")
        for x in range(1000)
    ]


def listRandinsTourLength():
    return [
        readTourLength(
            "Workspace/TSP/Data/fullRandom_large_" + str(x) + "_randins.json"
        )
        for x in range(1000)
    ]


def listNearinsTourLength():
    return [
        readTourLength(
            "Workspace/TSP/Data/fullRandom_large_" + str(x) + "_nearins.json"
        )
        for x in range(1000)
    ]


def listCheapinsTourLength():
    return [
        readTourLength(
            "Workspace/TSP/Data/fullRandom_large_" + str(x) + "_cheapins.json"
        )
        for x in range(1000)
    ]


def listMSTTourLength():
    return [
        readTourLength("Workspace/TSP/Data/fullRandom_large_" + str(x) + "_mst.json")
        for x in range(1000)
    ]


def listChristoTourLength():
    return [
        readTourLength(
            "Workspace/TSP/Data/fullRandom_large_" + str(x) + "_christo.json"
        )
        for x in range(1000)
    ]


def listOptimalTourLength():
    return [
        readTourLength("Workspace/TSP/Data/fullRandom_large_" + str(x) + "_sol.json")
        for x in range(1000)
    ]


def listGreedyTour():
    return [
        readTour("Workspace/TSP/Data/fullRandom_large_" + str(x) + "_greedy.json")
        for x in range(1000)
    ]


def listNearnTour():
    return [
        readTour("Workspace/TSP/Data/fullRandom_large_" + str(x) + "_nearn.json")
        for x in range(1000)
    ]


def listFarinsTour():
    return [
        readTour("Workspace/TSP/Data/fullRandom_large_" + str(x) + "_farins.json")
        for x in range(1000)
    ]


def listRandinsTour():
    return [
        readTour("Workspace/TSP/Data/fullRandom_large_" + str(x) + "_randins.json")
        for x in range(1000)
    ]


def listNearinsTour():
    return [
        readTour("Workspace/TSP/Data/fullRandom_large_" + str(x) + "_nearins.json")
        for x in range(1000)
    ]


def listCheapinsTour():
    return [
        readTour("Workspace/TSP/Data/fullRandom_large_" + str(x) + "_cheapins.json")
        for x in range(1000)
    ]


def listMSTTour():
    return [
        readTour("Workspace/TSP/Data/fullRandom_large_" + str(x) + "_mst.json")
        for x in range(1000)
    ]


def listChristoTour():
    return [
        readTour("Workspace/TSP/Data/fullRandom_large_" + str(x) + "_christo.json")
        for x in range(1000)
    ]


def listOptimalTour():
    return [
        readTour("Workspace/TSP/Data/fullRandom_large_" + str(x) + "_sol.json")
        for x in range(1000)
    ]


def listToTuple(list):
    newList = []

    for i in range(len(list)):
        newList.append([])
        for j in range(len(list[i]) - 1):
            newList[i].append((list[i][j], list[i][j + 1]))
    return newList
