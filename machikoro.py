import numpy as np
from numpy import random
import random as r
import time
import itertools as iter
from tqdm import tqdm
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
            state[3][action-1] -= 1
        elif action in np.arange(16,20):#upgrade
            state[0][player] -= self.card_costs[action-1]
            state[2][player][action-16] = 1
        elif action in np.arange(20,22):#if this action is chosen the current player has another action except stealing
            #stealing 5 coins
            #20/21/22 means steal 5 coins from player curr_player+1, curr_player+2, curr_player+3
            coins = state[0][(player+(action-19))%len(state[0])]
            state[0][player] = state[0][player]+coins if coins < 5 else state[0][player]+5
            state[0][(player+(action-19))%len(state[0])] = 0 if coins <5 else state[0][(player+(action-19))%len(state[0])]-5 
            return state
        return state
    
    def distribution(self,state,player,dice):
        current_player = player 
        dice_sum = np.sum(dice)
        #order of distribution
        #active player first makes his payments bc of others red cards then all players collect their coins then active player may take from other players
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

    def get_valid_actions(self,state,player,can_steal = True):
        #first check which cards are available, afterwards filter the too expensive ones #regard upgrade actions stealing 5 coins or swapping cards which are handled before the card option
        num_coins = state[0][player]
        available_cards = (state[3]>0)
        for n in range(len(available_cards)):
            if n not in [6,7,8]:
                available_cards[n] =  available_cards[n] and (num_coins >= self.card_costs[n])
            else:
                available_cards[n] = (state[1][player][n] == 0) and (num_coins >= self.card_costs[n])
                #TRADING DISABLED bc of too high complexity
                if n == 8:
                    available_cards[n] = False

        #available_cards contains all buyable cards // all buy card actions + action 0 ~ do nothing and actions 16,17,18,19 for upgrade option
        available_cards = np.append(np.array([]),np.argwhere(np.append(np.array([True]),available_cards)))

        upgrade_cost = np.array([4,10,16,22])
        upgrade_index = np.array([16,17,18,19])
        for i in range(len(upgrade_cost)):
            if num_coins >= upgrade_cost[i] and state[2][player][i] == 0:
                available_cards = np.append(available_cards,upgrade_index[i])

        steal_index = np.array([20,21,22])
        if state[1][player][7] and can_steal:
            available_cards = np.append(available_cards,steal_index[:len(state[0])-1])
        return available_cards.astype(int)
    
    #Checks wether any player has all upgrades and resturns who has all upgrades is_terminated(state)[1] contains the value for if the game has ended
    #unlocked all upgrades used to determine the winner
    def is_terminated(self,state):
        unlocked_all_upgrades = [np.all(state[2][i]) for i in range(len(state[2]))]
        return unlocked_all_upgrades,np.any(unlocked_all_upgrades)

    def get_expected_reward(self,state,player,two_dice=0):
        xR = np.zeros(len(state[0]))
        dice_values = np.arange(2,13) if two_dice==1 else np.arange(1,7)
        probs = dice_probs if two_dice==1 else np.zeros(12) + 1/6
        for val in dice_values:
            state_copy = copy.deepcopy(state)
            if val == 6 and state[1][player][7]: #if 5 coins might be stolen subtract the expected stolen value being own_coins/sum_coins bc the more coins one has the more likely the coins are being stolen
                collective_coins = np.sum(state[0])-state[0][player] 
                for i in range(len(state[0])):
                    if i != player and collective_coins>0:
                        xR[i] = xR[i]- (1/6 * 5 * (state[0][i] / collective_coins))
            self.distribution(state_copy,player,val)
            xR = xR + ((state_copy[0] - state[0])*probs[val])    
        return xR

    def expected_reward_after_one_roatation(self,state):
        xR = np.zeros(len(state[0]))
        for i in range(len(state[0])):
            reward = self.get_expected_reward(state,i)
            if state[2][i][0]:
                reward = (reward + self.get_expected_reward(state,i,1))/2
            xR = xR + reward
        return xR

dice_probs = {
            2 : 1/36,
            3 : 2/36,
            4 : 3/36,
            5 : 4/36,
            6 : 5/36,
            7 : 6/36,
            8 : 5/36,
            9 : 4/36,
            10: 3/36,
            11: 2/36,
            12: 1/36,
        } 
#throws always two dice
def dice():
    return random.randint(1,7,size=(2))

def random_bot_action(valid_actions):
    return r.choice(valid_actions)

#greedy bot should use 1 or 2 dice based on wether the expected reward is higher or lower
def greedy_bot_action(valid_actions):#greedy bot tries to buy in every turn choosing randomly which specific card to buy prioritising upgrades 
    upgrade_index = np.array([16,17,18,19])
    upgradable = np.isin(upgrade_index,valid_actions)
    if np.any(upgradable):
        return upgrade_index[r.choice(np.argwhere(upgradable))[0]]
    if len(valid_actions)>1:
        return r.choice(valid_actions[valid_actions>0])   
    return 0

#this bot is better in 1v1s against the greedy bot but succumbs in a game of >2 players to the greedy bot
def expecting_greedy_bot_action(game,state,player,valid_actions):
    upgrade_index = np.array([16,17,18,19])
    upgradable = np.isin(upgrade_index,valid_actions)
    if np.any(upgradable):
        return upgrade_index[r.choice(np.argwhere(upgradable))[0]]
    if len(valid_actions)==1:
        return valid_actions[0]
    expected_value = np.zeros(len(valid_actions)-1)
    for v in range(1,len(valid_actions)):
        state_copy = copy.deepcopy(state)
        state_copy = game.get_next_state(state_copy,player,valid_actions[v])
        expected_value[v-1] = game.expected_reward_after_one_roatation(state_copy)[player]
    return valid_actions[np.argmax(expected_value)+1]

def gameloop(iterations,playerIds):
    x = np.array([np.zeros(len(playerIds))])
    round_count = np.array([0])
    for i in tqdm(range(iterations)):
        Machikoro = machikoro()
        state = Machikoro.get_initial_state(len(playerIds))
        player = 0
        round = 0
        repeated = False
        while not Machikoro.is_terminated(state)[1]:
            round += 1
            j = 1
            #UPGRADE 1 HANDLING
            if state[2][player][0]:#choose one or two dice
                if playerIds[player]>=1:    
                    a = Machikoro.get_expected_reward(state,player,1)[player]
                    b = Machikoro.get_expected_reward(state,player,2)[player]
                    j = 2 if b>a else 1
                else:
                    j = r.choice([1,2])
            ###################
            dice_throw = dice()[:j]
            #UPGRADE 4 HANDLING
            if state[2][player][3] and playerIds[player]>=1:#choose to rethrow random bot has no use of rethrow 
                state_copy = copy.deepcopy(state)
                coins_before_payout = state_copy[0][player]
                Machikoro.distribution(state_copy,player,dice_throw)
                new_coin_number = coins_before_payout - state_copy[0][player]
                if (len(dice_throw)==1 and new_coin_number < a) or (len(dice_throw)==2 and new_coin_number < b):
                    dice_throw = dice()[:j]
            ###################
            Machikoro.distribution(state,player,dice_throw)
            v = Machikoro.get_valid_actions(state,player)
            act = expecting_greedy_bot_action(Machikoro,state,player,v) if playerIds[player]==2 else greedy_bot_action(v) if playerIds[player]==1 else random_bot_action(v)
            state = Machikoro.get_next_state(state,player,act)
            if act in np.arange(20,23):
                v = Machikoro.get_valid_actions(state,player,False)
                act = expecting_greedy_bot_action(Machikoro,state,player,v) if playerIds[player]==2 else greedy_bot_action(v) if playerIds[player]==1 else random_bot_action(v)
                state = Machikoro.get_next_state(state,player,act)
            #UPGRADE 3 HANDLING
            if len(dice_throw)==2 and not repeated and dice_throw[0]==dice_throw[1]:#if the two used dice are identical the player gets another turn but only once
                repeated = True
            else:
                player = (player+1)%len(state[0])
                repeated = False
            ###################
        x = np.append(x,[np.sum(state[2],axis=1)],axis=0)
        round_count = np.append(round_count,round)
    return x[1:],round_count[1:]

# Machikoro = machikoro()
# state = Machikoro.get_initial_state(2)
# print(state)
# state[0][0] = 100
# print(state)
# v = Machikoro.get_valid_actions(state,0)
# print(v)
# print(expecting_greedy_bot_action(Machikoro,state,0,v))
st = time.process_time()
x,round_count = gameloop(1000,[2,2])
et = time.process_time()
res = et - st
print('CPU Execution time:', res, 'seconds')
print(np.average(x,axis=0))
print(np.median(x,axis=0))
print(np.max(round_count,axis=0))
print(np.min(round_count,axis=0))
print(np.average(round_count,axis=0))

#late game turn iteration
#select one or two dice
#throw dice
#choose to rethrow or not
#check for doublets (Pasch) for a new possible turn after the current
#distribute the earnings/coins for respective dice value
#if special card is owned -> trade cards/steal coins
#choose 1 or 0 of 15 cards OR 1 or 0 of big projects to buy/build
#end turn
