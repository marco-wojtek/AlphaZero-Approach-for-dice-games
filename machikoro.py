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
    
    def get_next_state(self,state,player,action,dice):#action is the choice if and which card is to buy #action 0 is always = "do nothing" 
        if action == -1:
            #basic action like buying a card
            return #TODO
        elif action in [16,17,18,19]:#upgrade
            state[0][player] = state[0][player]-4 if action == 16 else state[0][player]-10 if action == 17 else state[0][player]-16 if action==18 else state[0][player]-22
            state[2][action-15] = 1
        else:
            #stealing/trading
            return #TODO
        return state
    
    def distribution(self,state,player,dice):
        current_player = player 
        dice_sum = np.sum(dice)
        
        if dice_sum in [3,9,10]:#red cards
            current_player -= 1
            #counterclockwise payment iteration
            while current_player != player:
                #action with regards to upgrades
                coins = state[1][current_player][3] + (state[1][current_player][3]*state[2][current_player][1]) if dice_sum == 3 else (2*state[1][current_player][12]) + (state[1][current_player][12]*state[2][current_player][1])
                if state[0][player] > coins:
                    state[0][player] -= coins
                    state[0][current_player] += coins
                else: 
                    state[0][current_player] += state[0][player]
                    state[0][player] = 0
                current_player = current_player-1 if current_player>0 else (len(state[0])-1)
        if dice_sum in [1,2,5,9,10]:#blue cards
            #clockwise collecting iteration
            while True:
                #action with regards to upgrades
                if dice_sum == 1:
                    coins = state[1][current_player][0]
                elif dice_sum == 2:
                    coins = state[1][current_player][1]
                elif dice_sum == 5:
                    coins = state[1][current_player][5]
                elif dice_sum == 9:
                    coins = state[1][current_player][11] * 5
                else:
                    coins = state[1][current_player][13] * 3
                state[0][current_player] += coins
                current_player = (current_player+1)%len(state[0])
                if current_player == player:
                    break
        if dice_sum in [2,3,4,7,8,11,12]:#green cards
            if dice_sum in [2,3] :
                state[0][current_player] += state[1][current_player][2]
            elif dice_sum == 4:
                state[0][current_player] += state[1][current_player][4]*3
            elif dice_sum == 7:
                state[0][current_player] += state[1][current_player][9] * state[1][current_player][1] * 3
            elif dice_sum == 8:
                state[0][current_player] += state[1][current_player][10] * (state[1][current_player][5]+state[1][current_player][11]) * 3
            else:
                state[0][current_player] += state[1][current_player][14] * (state[1][current_player][0]+state[1][current_player][13]) * 2
        if dice_sum == 6:
            current_player += 1
            while True and state[1][player][6]>0:
                if state[0][current_player] >= 2:
                    state[0][current_player] -= 2
                    state[0][player] += 2
                else:
                    state[0][player] += state[0][current_player]
                    state[0][current_player] = 0
                current_player = (current_player+1)%len(state[0])
                if current_player == player:
                    break

    def valid_actions(self,state,player):
        #first check which cards are available, afterwards filter the too expensive ones #regard upgrade actions stealing 5 coins or swapping cards which are handled before the card option
        num_coins = state[0][player]
        available_cards = (state[3]>0)
        available_cards[0]  = available_cards[0] and num_coins >= 1
        available_cards[1]  = available_cards[1] and num_coins >= 1
        available_cards[2]  = available_cards[2] and num_coins >= 1
        available_cards[3]  = available_cards[3] and num_coins >= 2
        available_cards[4]  = available_cards[4] and num_coins >= 2
        available_cards[5]  = available_cards[5] and num_coins >= 3
        available_cards[6]  = (state[1][player][6] == 0) and num_coins >= 6
        available_cards[7]  = (state[1][player][7] == 0) and num_coins >= 7
        available_cards[8]  = (state[1][player][8] == 0) and num_coins >= 8
        available_cards[9]  = available_cards[9]  and num_coins >= 5
        available_cards[10] = available_cards[10] and num_coins >= 3
        available_cards[11] = available_cards[11] and num_coins >= 6
        available_cards[12] = available_cards[12] and num_coins >= 3
        available_cards[13] = available_cards[13] and num_coins >= 3
        available_cards[14] = available_cards[14] and num_coins >= 2
        #available_cards contains all buyable cards // all buy card actions + action 0 ~ do nothing and actions 16,17,18,19 for upgrade option
        available_cards = np.append(np.array([]),np.argwhere(np.append(np.array([True]),available_cards)))
        if num_coins >= 22:
            available_cards = np.append(available_cards,np.array([16,17,18,19]))
        elif num_coins >= 16:
            available_cards = np.append(available_cards,np.array([16,17,18]))
        elif num_coins >= 10:
            available_cards = np.append(available_cards,np.array([16,17]))
        elif num_coins >= 4:
            available_cards = np.append(available_cards,np.array([16]))
        return available_cards
    
    #Checks wether any player has all upgrades
    def is_terminated(self,state):
        return np.any([np.all(state[2][i]) for i in range(len(state[2]))])

#throws always two dice, index 0 is the relevant one for one dice throws
def dice():
    return random.randint(1,7,size=(2))

Machikoro = machikoro()


#late game turn iteration
#select one or two dice
#throw dice
#choose to rethrow or not
#check for doublets (Pasch) for a new possible turn after the current
#distribute the earnings/coins for respective dice value
#if special card is owned -> trade cards/steal coins
#choose 1 or 0 of 15 cards OR 1 or 0 of big projects to buy/build
#end turn