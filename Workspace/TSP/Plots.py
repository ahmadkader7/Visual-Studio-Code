import matplotlib.pyplot as plt


def plotter(tuple1, tuple2, name):

    fig, ax = plt.subplots(2, 2)

    ax[0][0].hist(tuple1, bins=16, edgecolor='black')
    ax[0][1].boxplot(tuple1, showmeans=True)
    ax[1][0].hist(tuple2, bins=16, edgecolor='black')
    ax[1][1].boxplot(tuple2, showmeans=True)

    ax[0][0].set_title(
        'How many optimal solutions edges are inside the ' + name + '?')
    ax[0][0].set_xlabel('optimal solutions edges in ' + name + ' in percent')
    ax[0][0].set_ylabel('amount of instances')

    ax[0][1].set_title(
        'How many optimal solutions edges are inside the ' + name + '?')
    ax[0][1].set_ylabel('optimal solutions edges in ' + name + ' in percent')

    ax[1][0].set_title(
        'How many greedy solutions edges are inside the ' + name + '?')
    ax[1][0].set_xlabel('greedy solutions edges in ' + name + ' in percent')
    ax[1][0].set_ylabel('amount of instances')

    ax[1][1].set_title(
        'How many greedy solutions edges are inside the ' + name + '?')
    ax[1][1].set_ylabel('greedy solutions edges in ' + name + ' in percent')

    plt.show()


def scatterPlotter(approximations, percents):
    colors = ['green', 'red', 'blue', 'purple',
              'pink', 'yellow', 'cyan', 'grey', 'black']
    for i in range(len(approximations)):
        plt.scatter(approximations[i], percents[i],
                    color=colors[i], edgecolors='black')
    plt.title(
        'comparison of the amount of intersections of edges with the optimal solution')
    plt.xlabel('quality of approximation')
    plt.ylabel('percentage of intersetions')
    plt.legend(['greedy', 'nearest neighbor', 'farthest insertion', 'random insertion', 'nearest insertion',
               'cheapest insertion', 'mst heuristic', 'christofides', 'minimum spanning tree'])
    plt.show()
