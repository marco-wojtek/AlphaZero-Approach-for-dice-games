import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import random as r


def mcts_simulation_against_rdmBot(game, num_searches, num_games, version):
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
 
def loss_img(game_idx,NN_id,x):
    game = ["Yahtzee", "Machikoro", "Quixx"]
    a = ["total","policy","value"]
    for loss in range(2,5):
        file = open(f"{game[game_idx]}/LossesNN{NN_id}/Losses{loss}/{a[x]}_loss.txt", "r")
        lines = np.array(file.read().split(),dtype=float)
        plt.plot(np.arange(1,len(lines)+1), lines,marker='o',label = f"lr = {10**(-loss)}")
        file.close()
    plt.title(f'{a[x]} loss over 8 generations of 6 epochs')
    plt.xlabel('Generation')
    plt.ylabel('loss')
    plt.xticks(np.arange(1,len(lines)+7,step=6))
    plt.grid(axis='x')
    plt.legend(loc="lower right")
    plt.show()
#loss_img(1,2,2)
#open('policy_loss.txt', 'w').close()