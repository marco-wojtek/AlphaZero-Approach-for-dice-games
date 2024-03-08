import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import random as r

# from Yahtzee import Y_NN
# from Quixx import Q_NN
# from MachiKoro import M_NN

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
 
#simulates games between a model against a random_bot; each player starts the same number of games
def model_simulation_against_rdmBot(game, num_games, version):
    winrates = np.array([])
    iteration = np.arange(8)
    for iter in iteration:
        wr, ties = game.simulate(num_games,iter,None,version)
        winrates = np.append(winrates,wr)

    plt.plot(iteration, winrates)

#tests model iterations against each other
def model_simulation(game, num_games, version,m1,m2):
    wr, ties = game.simulate(num_games,iter,None,version)

def loss_image():
    return
# for i in range(2,5):
#     model_simulation_against_rdmBot(Y_NN,1000,i)
# plt.title('Win rate against random Bot')
# plt.xlabel('Iteration')
# plt.ylabel('Win rate in %')
# plt.show()

x = np.arange(1,49)
y = [r.randint(0,100) for i in range(len(x))]
plt.plot(x,y,marker='o')
plt.xticks(np.arange(1,49,step=6))
plt.grid(axis='x')
plt.show()