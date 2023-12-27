import numpy as np
from numpy import random
import random as r
import time
import itertools as iter
import copy

class machikoro:
    
    def __init__(self):
        self.card_costs = np.array([1,1,1,2,2,3,6,7,8,5,3,6,3,3,2,4,10,16,22])#last 4 are upgrade costs

    def get_initial_state(self,num_players):
        player_bank = np.zeros(num_players) + 3
        player_cards = np.zeros((num_players,15))
        player_cards[:,0:2] = 1
        player_upgrades = np.zeros((num_players,4))
        game_board = np.zeros(15) + 6
        game_board[6:9] = 4
        return np.array([player_bank,player_cards,player_upgrades,game_board],dtype=object)
    
    def get_next_state(self,state,player,action):#action is the choice if and which card is to buy #action 0 is always = "do nothing" 
        if action in np.arange(1,16):
            #basic action like buying a card
            state[0][player] -= self.card_costs[action-1]
            state[1][player][action-1] += 1 
        elif action in [16,17,18,19]:#upgrade
            state[0][player] -= self.card_costs[action-1]
            state[2][player][action-16] = 1
        elif action > 19:
            #stealing/trading
            return state
        return state
    
    def distribution(self,state,player,dice):
        current_player = player 
        dice_sum = np.sum(dice)
        
        if dice_sum in [3,9,10]:#red cards
            current_player = current_player-1 if current_player>0 else (len(state[0])-1)
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
            current_player = (current_player+1)%len(state[0])
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

    def get_valid_actions(self,state,player):
        #first check which cards are available, afterwards filter the too expensive ones #regard upgrade actions stealing 5 coins or swapping cards which are handled before the card option
        num_coins = state[0][player]
        available_cards = (state[3]>0)
        for n in range(len(available_cards)):
            if n not in [6,7,8]:
                available_cards[n] =  available_cards[n] and (num_coins >= self.card_costs[n])
            else:
                available_cards[n] = (state[1][player][n] == 0) and (num_coins >= self.card_costs[n])
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
    
    #Checks wether any player has all upgrades and resturns who has all upgrades is_terminated(state)[1] contains the value for if the game has ended
    #unlocked all upgrades used to determine the winner
    def is_terminated(self,state):
        unlocked_all_upgrades = [np.all(state[2][i]) for i in range(len(state[2]))]
        return unlocked_all_upgrades,np.any(unlocked_all_upgrades)

#throws always two dice, index 0 is the relevant one for one dice throws
def dice():
    return random.randint(1,7,size=(2))

Machikoro = machikoro()
initial = Machikoro.get_initial_state(2)
c = copy.deepcopy(initial)


#late game turn iteration
#select one or two dice
#throw dice
#choose to rethrow or not
#check for doublets (Pasch) for a new possible turn after the current
#distribute the earnings/coins for respective dice value
#if special card is owned -> trade cards/steal coins
#choose 1 or 0 of 15 cards OR 1 or 0 of big projects to buy/build
#end turn

#TODO: Add the stealing and trading actions 