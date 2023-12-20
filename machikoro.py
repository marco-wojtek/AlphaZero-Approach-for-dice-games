import numpy as np
from numpy import random
import random as r
import time
import itertools as iter

# action space of size 20: 1 for no action 15 for all buyable cards 4 unlockable or usable upgrades (if an upgrade is unlocked using the action space makes use of the effect)
# cards have attributes: cost/value, reward/action_reward, reward_condition (needed dice value for reward)
# each player has a card collection (maybe as dictionary and a libary for all cards) and money stack (maybe as stack depending on how easy the empty stack handle is)
# dice don't need a seperate class since only 1/2 dice are possible

class card:
    #categories: 
    # 1: blue -> raw materials industry
    # 2: green -> store, factory, market hall
    # 3: red -> café, restaurant
    # 4: violet -> special buisness
    # 0: yellow -> large-scale project alias UPGRADES
    def __init__(self,cost,reward_condition,category):
        self.cost = cost
        self.reward_condition = reward_condition
        assert category in [0,1,2,3,4]
        self.category = category
        self.can_trade = False if self.category in [0,4] else True
        self.every_turn = True if self.category in [1,3] else False
     
class machikoro:
    
    def get_initial_state(self,num_players):
        player_bank = np.zeros(num_players) + 3
        player_cards = np.zeros((num_players,15))
        player_cards[:,0:2] = 1
        player_upgrades = np.zeros((num_players,4))
        game_board = np.zeros(15) + 6
        game_board[6:9] = 4
        return np.array([player_bank,player_cards,player_upgrades,game_board],dtype=object)
    
    def get_next_state(self,state,player,action,dice):#action is the choice if and which card is to buy if -1 it begins the coin distribution
        if action == -1:
            self.distribution(state,player,dice)
        elif action in [15,16,17,18]:#upgrade
            state[0][player] = state[0][player]-4 if action == 15 else state[0][player]-10 if action == 16 else state[0][player]-16 if action==17 else state[0][player]-22
            state[2][action-15] = 1
        else:
            state[0][player] = state[0][player] if action in [] else state[0][player] 
            state[1][action] += 1
            state[3][action] -= 1
        return state
    
    def distribution(self,state,player,dice):
        current_player = player -1 
        #counterclockwise payment iteration
        while current_player != player:
            #action with regards to upgrades

            current_player = current_player-1 if current_player>0 else len(state[0])
        #clockwise collecting iteration
        while True:
            #action with regards to upgrades
            
            current_player = (current_player+1)%len(state[0])
            if current_player == player:
                break

    def valid_actions(self,state,player,dice):
        #first check which cards are available, afterwards filter the too expensive ones #regard upgrade actions stealing 5 coins or swapping cards which are handled before the card option
        return 0
    
    #Checks wether any player has all upgrades
    def is_terminated(self,state):
        return np.any([np.all(state[2][i]) for i in range(len(state[2]))])

Machikoro = machikoro()
initial = Machikoro.get_initial_state(4)
print(initial[2])
initial[2][1] = 1
print([np.all(initial[2][i]) for i in range(len(initial))])
print(np.any([np.all(initial[2][i]) for i in range(len(initial[2]))]))