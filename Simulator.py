import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from Yahtzee import y
from Quixx import simpleQ as q
from MachiKoro import machikoro as m
    
def simulation(game, num_searches, num_games):
    winrates = np.array([])
    searches = np.arange(int(num_searches/10), num_searches, step= int(num_searches/10))

    for search_num in searches:
        winrate = game.simulate(search_num, num_games)

        winrate = np.append(winrates,winrate)

    plt.plot(searches, winrates)
    plt.title('Win rate against random Bot')
    plt.xlabel('MCTS searches')
    plt.ylabel('Win rate in %')

    plt.show()

   