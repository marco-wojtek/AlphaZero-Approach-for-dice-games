import numpy as np
from numpy import random
import random as r
import time
import itertools as iter
from tqdm import tqdm
import copy
import math

class Machikoro:
    
    def __init__(self):
        self.card_costs = np.array([1,1,1,2,2,3,6,7,8,5,3,6,3,3,2,4,10,16,22])#last 4 are upgrade costs

    def get_initial_state(self,num_players):
        player_bank = np.zeros(num_players,dtype=int) + 3
        player_cards = np.zeros((num_players,15),dtype=int)
        player_cards[:,0] = 1
        player_cards[:,2] = 1
        player_upgrades = np.zeros((num_players,4),dtype=int)
        game_board = np.zeros(15,dtype=int) + 6
        game_board[6:9] = num_players
        #disable trading and stealing 5
        game_board[7:9] = 0
        return np.array([np.zeros(2,dtype=int),player_bank,player_cards,player_upgrades,game_board],dtype=object)
    
    def get_next_state(self,state,player,action):#action is the choice if and which card is to buy #action 0 is always = "do nothing" 
        if action in np.arange(1,16):
            #basic action like buying a card
            state[1][player] -= self.card_costs[action-1]
            state[2][player][action-1] += 1 
            state[4][action-1] -= 1
        elif action in np.arange(16,20):#upgrade
            state[1][player] -= self.card_costs[action-1]
            state[3][player][action-16] = 1
        elif action in np.arange(20,22):#if this action is chosen the current player has another action except stealing
            #stealing 5 coins
            #20/21/22 means steal 5 coins from player curr_player+1, curr_player+2, curr_player+3
            coins = state[1][(player+(action-19))%len(state[1])]
            state[1][player] = state[1][player]+coins if coins < 5 else state[1][player]+5
            state[1][(player+(action-19))%len(state[1])] = 0 if coins <5 else state[1][(player+(action-19))%len(state[1])]-5 
            return state
        return state
    
    def distribution(self,state,player):
        current_player = player 
        dice_sum = np.sum(state[0])
        #order of distribution
        #active player first makes his payments bc of others red cards then all players collect their coins then active player may take from other players
        if dice_sum in [3,9,10]:#red cards
            current_player = current_player-1 if current_player>0 else (len(state[1])-1)
            #counterclockwise payment iteration
            while current_player != player:
                #action with regards to upgrades
                coins = state[2][current_player][3] + (state[2][current_player][3]*state[3][current_player][1]) if dice_sum == 3 else (2*state[2][current_player][12]) + (state[2][current_player][12]*state[3][current_player][1])
                if state[1][player] > coins:
                    state[1][player] -= coins
                    state[1][current_player] += coins
                else: 
                    state[1][current_player] += state[1][player]
                    state[1][player] = 0
                current_player = current_player-1 if current_player>0 else (len(state[1])-1)
        if dice_sum in [1,2,5,9,10]:#blue cards
            #clockwise collecting iteration
            while True:
                #action with regards to upgrades
                if dice_sum == 1:
                    coins = state[2][current_player][0]
                elif dice_sum == 2:
                    coins = state[2][current_player][1]
                elif dice_sum == 5:
                    coins = state[2][current_player][5]
                elif dice_sum == 9:
                    coins = state[2][current_player][11] * 5
                else:
                    coins = state[2][current_player][13] * 3
                state[1][current_player] += coins
                current_player = (current_player+1)%len(state[1])
                if current_player == player:
                    break
        if dice_sum in [2,3,4,7,8,11,12]:#green cards
            if dice_sum in [2,3] :
                state[1][current_player] += state[2][current_player][2]
            elif dice_sum == 4:
                state[1][current_player] += state[2][current_player][4]*3
            elif dice_sum == 7:
                state[1][current_player] += state[2][current_player][9] * state[2][current_player][1] * 3
            elif dice_sum == 8:
                state[1][current_player] += state[2][current_player][10] * (state[2][current_player][5]+state[2][current_player][11]) * 3
            else:
                state[1][current_player] += state[2][current_player][14] * (state[2][current_player][0]+state[2][current_player][13]) * 2
        if dice_sum == 6:#buffed take two
            current_player = (current_player+1)%len(state[1])
            while True and state[2][player][6]>0:
                if state[1][current_player] >= 3:
                    state[1][current_player] -= 3
                    state[1][player] += 3
                else:
                    state[1][player] += state[1][current_player]
                    state[1][current_player] = 0
                current_player = (current_player+1)%len(state[1])
                if current_player == player:
                    break
        #limiter for coin stack        
        for n in range(len(state[1])):
            if state[1][n] > 63:
                state[1][n] = 63

    def get_valid_moves(self,state,player,can_steal = False):#active Stealing disabled for 2 players bc coins are always taken from opponent
        #first check which cards are available, afterwards filter the too expensive ones #regard upgrade actions stealing 5 coins or swapping cards which are handled before the card option
        num_coins = state[1][player]
        available_cards = (state[4]>0)
        for n in range(len(available_cards)):
            if n not in [6,7,8]:
                available_cards[n] =  available_cards[n] and (num_coins >= self.card_costs[n])
            else:
                available_cards[n] = (state[2][player][n] == 0) and (num_coins >= self.card_costs[n])
                #TRADING DISABLED bc of too high complexity
                if n == 8 or n == 7:
                    available_cards[n] = False

        #available_cards contains all buyable cards // all buy card actions + action 0 ~ do nothing and actions 16,17,18,19 for upgrade option
        available_cards = np.append(np.array([]),np.argwhere(np.append(np.array([True]),available_cards)))

        upgrade_cost = np.array([4,10,16,22])
        upgrade_index = np.array([16,17,18,19])
        for i in range(len(upgrade_cost)):
            if num_coins >= upgrade_cost[i] and state[3][player][i] == 0:
                available_cards = np.append(available_cards,upgrade_index[i])

        steal_index = np.array([20,21,22])
        if state[2][player][7] and can_steal:
            available_cards = np.append(available_cards,steal_index[:len(state[1])-1])
        return available_cards.astype(int)
    
    def is_terminated(self,state):
        unlocked_all_upgrades = [np.all(state[3][i]) for i in range(len(state[3]))]
        terminal = np.any(unlocked_all_upgrades) 
        winner = np.argmax(unlocked_all_upgrades) if terminal else -1
        return winner, terminal

    def get_expected_reward(self,state,player,two_dice=0):
        xR = np.zeros(len(state[1]))
        dice_values = np.arange(2,13) if two_dice==1 else np.arange(1,7)
        probs = dice_probs if two_dice==1 else np.zeros(12) + 1/6
        for val in dice_values:
            state_copy = copy.deepcopy(state)
            if val == 6 and state[2][player][7]: #if 5 coins might be stolen subtract the expected stolen value being own_coins/sum_coins bc the more coins one has the more likely the coins are being stolen
                collective_coins = np.sum(state[1])-state[1][player] 
                for i in range(len(state[1])):
                    if i != player and collective_coins>0:
                        xR[i] = xR[i]- (1/6 * 5 * (state[1][i] / collective_coins))
            self.distribution(state_copy,player,val)
            xR = xR + ((state_copy[0] - state[1])*probs[val])    
        return xR

    def expected_reward_after_one_roatation(self,state):
        xR = np.zeros(len(state[1]))
        for i in range(len(state[1])):
            reward = self.get_expected_reward(state,i)
            if state[3][i][0]:
                reward = (reward + self.get_expected_reward(state,i,1))/2
            xR = xR + reward
        return xR
    
    def get_encoded_state(self,state):# encoded as P0 Coins, P1 Coins, P0 cards+upgrades, P1 cards+upgrades, Cards on gameboard
        encoded = np.array([])
        encoded = np.append(encoded,get_one_hot(state[0][0]+1,7))
        encoded = np.append(encoded,get_one_hot(state[0][1]+1,7))
        for i in range(len(state[1])):
            encoded = np.append(encoded,get_binary(state[1][i]))

        for i in range(len(state[1])):
            for card in state[2][i]:
                encoded = np.append(encoded,get_one_hot(card+1,8))
            #encoded = np.append(encoded,state[2][i]>0)
            encoded = np.append(encoded,state[3][i])
        #new
        for card in state[4]:
            encoded = np.append(encoded,get_one_hot(card+1,7))
        #return np.append(encoded,state[4]>0) #old
        return encoded
    
    def get_encoded_states(self,states):
        stack = np.array([self.get_encoded_state(states[0])])
        for i in range(1,len(states)):
            st = states[i]
            stack = np.append(stack,[self.get_encoded_state(st)],axis=0)
        return stack

def get_one_hot(num,size):
    one_hot = np.zeros(size)
    one_hot[int(size-num)] = 1
    return one_hot

def get_binary(num):
    if num>= 64:
        num = 63
    return np.array(list(np.binary_repr(int(num),width=6))).astype(int)           

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
def dice(size):
    d = random.randint(1,7,size=(2))
    if size == 1:
        d[1] = 0
    elif size == 0:
        d[:] = 0
    return d

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
        machikoro = Machikoro()
        state = Machikoro.get_initial_state(len(playerIds))
        player = 0
        round = 0
        repeated = False
        while not Machikoro.is_terminated(state)[1]:
            round += 1
            j = 1
            #UPGRADE 1 HANDLING
            if state[3][player][0]:#choose one or two dice
                if playerIds[player]>=1:    
                    a = Machikoro.get_expected_reward(state,player,1)[player]
                    b = Machikoro.get_expected_reward(state,player,2)[player]
                    j = 2 if b>a else 1
                else:
                    j = r.choice([1,2])
            ###################
            dice_throw = dice()[:j]
            #UPGRADE 4 HANDLING
            if state[3][player][3] and playerIds[player]>=1:#choose to rethrow random bot has no use of rethrow 
                state_copy = copy.deepcopy(state)
                coins_before_payout = state_copy[0][player]
                Machikoro.distribution(state_copy,player,dice_throw)
                new_coin_number = coins_before_payout - state_copy[0][player]
                if (len(dice_throw)==1 and new_coin_number < a) or (len(dice_throw)==2 and new_coin_number < b):
                    dice_throw = dice()[:j]
            ###################
            Machikoro.distribution(state,player,dice_throw)
            v = Machikoro.get_valid_moves(state,player)
            act = expecting_greedy_bot_action(Machikoro,state,player,v) if playerIds[player]==2 else greedy_bot_action(v) if playerIds[player]==1 else random_bot_action(v)
            state = Machikoro.get_next_state(state,player,act)
            if act in np.arange(20,23):
                v = Machikoro.get_valid_moves(state,player,False)
                act = expecting_greedy_bot_action(Machikoro,state,player,v) if playerIds[player]==2 else greedy_bot_action(v) if playerIds[player]==1 else random_bot_action(v)
                state = Machikoro.get_next_state(state,player,act)
            #UPGRADE 3 HANDLING
            if len(dice_throw)==2 and not repeated and dice_throw[0]==dice_throw[1]:#if the two used dice are identical the player gets another turn but only once
                repeated = True
            else:
                player = (player+1)%len(state[1])
                repeated = False
            ###################
        x = np.append(x,[np.sum(state[3],axis=1)],axis=0)
        round_count = np.append(round_count,round)
    return x[1:],round_count[1:]

# v = Machikoro.get_valid_moves(state,0)
# print(v)
# print(expecting_greedy_bot_action(Machikoro,state,0,v))
# st = time.process_time()
# x,round_count = gameloop(1000,[2,2])
# et = time.process_time()
# res = et - st
# print('CPU Execution time:', res, 'seconds')
# print(np.average(x,axis=0))
# print(np.median(x,axis=0))
# print(np.max(round_count,axis=0))
# print(np.min(round_count,axis=0))
# print(np.average(round_count,axis=0))

#late game turn iteration
#select one or two dice
#throw dice
#choose to rethrow or not
#check for doublets (Pasch) for a new possible turn after the current
#distribute the earnings/coins for respective dice value
#if special card is owned -> trade cards/steal coins
#choose 1 or 0 of 15 cards OR 1 or 0 of big projects to buy/build
#end turn
 
class Node:

    def __init__(self, game, args, state, active_player, parent=None, action_taken=None, ischance = False, isdice_node = False, isrethrow_node=False, num_of_dice = 0, dices = None):
        self.game = game
        self.state = state
        self.args = args
        self.active_player = active_player
        self.parent = parent
        self.children = {}

        self.action_taken = action_taken
        self.ischance = ischance
        self.isrethrow_node = isrethrow_node
        self.isdice_node = isdice_node
        self.num_of_dice = num_of_dice
        self.dices = dices
        dice_node = np.array([1,2]) if self.state[3][self.active_player][0] else np.array([1])
        rethrow_node = np.array([0,1]) if self.state[3][self.active_player][3] else np.array([0])
        self.expandable_moves = calc_dice_state_probabilities(self.num_of_dice) if self.ischance else dice_node if isdice_node else rethrow_node if isrethrow_node else game.get_valid_moves(state,active_player)
        self.visit_count = 0
        self.value_sum = 0

    def is_fully_expanded(self):
        if self.ischance:
            return len(self.expandable_moves) == len(self.children) and len(self.children) > 0
        
        return len(self.expandable_moves) == len(self.children) and len(self.children) > 0
    
    def select(self):
        if self.ischance:
            dsp = self.expandable_moves
            outcome = r.choices(list(dsp.keys()),list(dsp.values()))[0]
            return self.children[outcome]
        
        best_child = None
        best_ucb = -np.inf
        
        for child in self.children.values():
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb
                
        return best_child
    
    def get_ucb(self,child):
        #q_value is normalised from [-1,1] 
        if self.active_player==child.active_player:
            q_value = ((child.value_sum / child.visit_count) + 1) / 2
        else:#if the child has a different active player the q value is inverted because the score is from a different view
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.args['C'] * math.sqrt(math.log(self.visit_count) / child.visit_count)
    
    def expand(self):
        if self.ischance:
            for dices in self.expandable_moves.keys():
                child_state = copy.deepcopy(self.state)
                can_rethrow = self.state[3][self.active_player][3] and (self.parent is None or self.parent.isrethrow_node == False)
                if not can_rethrow:#if no rethrow is possible the state distributes based on the dices
                    self.game.distribution(child_state,self.active_player,np.array([int(x) for x in dices]))
                child = Node(self.game,self.args,child_state,self.active_player,self,None,False,False,can_rethrow,len(dices),dices)
                self.children[dices] = child
        elif self.isrethrow_node:
            index = r.choice(np.where(self.expandable_moves!=-1)[0])
            rethrow = self.expandable_moves[index]
            child_state = copy.deepcopy(self.state)
            if rethrow:#choosing to rethrow appends a chance node
                child = Node(self.game, self.args,child_state,self.active_player,self,rethrow,True, False, False, self.num_of_dice)
                child.expand()
            else:#else distribute with dices 
                self.game.distribution(child_state,self.active_player,np.array([int(x) for x in self.dices]))
                child = Node(self.game, self.args,child_state,self.active_player,self,rethrow,False, False, False, self.num_of_dice)
            self.children[rethrow] = child
            self.expandable_moves[index] = -1
        elif self.isdice_node:
            index = r.choice(np.where(self.expandable_moves!=-1)[0])
            num_of_dice = self.expandable_moves[index]
            child_state = copy.deepcopy(self.state)
            child = Node(self.game,self.args,child_state,self.active_player,self,None,True,False,False,num_of_dice)
            self.children[num_of_dice] = child
            child.expand()
            self.expandable_moves[index] = -1
        else:
            index = r.choice(np.where(self.expandable_moves!=-1)[0])
            action = self.expandable_moves[index]
            child_state = self.game.get_next_state(copy.deepcopy(self.state),self.active_player,action)
            child = Node(self.game,self.args,child_state,(self.active_player+1)%len(child_state[1]),self,action,False,True,False,0)
            self.children[action] = child
            self.expandable_moves[index] = -1

        return child
    
    def simulate(self):
        rollout_state = copy.deepcopy(self.state)
        player = self.active_player
        action = self.action_taken
        dice_choice = None
        #if the players do nothing for 6 turns concecuatively the game is terminated without winner
        action_cnt = 0
        #action block for expansion decision
        assert not self.parent.ischance
        if self.parent.isdice_node or self.parent.isrethrow_node:
            dice_choice = self.num_of_dice
        #
        #rest of the simulation
        #loop
        winner, is_terminal = self.game.is_terminated(rollout_state)
        while not is_terminal:
            #choose number of dice
            d = dice(dice_choice) if dice_choice is not None else (dice(1) if not rollout_state[3][player][0] else dice(r.choice([1,2])))
            #choose to rethrow will not be simulated due to random playout making it unnecissary
            #distribution
            self.game.distribution(rollout_state,player,d)
            #buy/skip action
            v = self.game.get_valid_moves(rollout_state,player,action not in np.arange(20,22))
            action = r.choice(v)
            rollout_state = self.game.get_next_state(rollout_state,player,action)
            #change players turn, unless doublets
            if not rollout_state[3][player][2] or (len(d) == 2 and d[0] != d[1]):
                player = (player+1)%len(rollout_state[1])
                dice_choice = None
            else:
                dice_choice = len(d)

            winner, is_terminal = self.game.is_terminated(rollout_state) 
            if action == 0:
                action_cnt += 1
                if action_cnt == 10:
                    #print("Simulation break after 10 eventless turns!")
                    break
            else:
                action_cnt = 0
        return winner
    
    def backpropagate(self,value):
        self.value_sum += (-1)**(value!=self.active_player) * (value>=0) 

        self.visit_count += 1
         
        if self.parent is not None:
            self.parent.backpropagate(value) 

def calc_dice_state_probabilities(num_of_dice):
    all_possible_dice_states = list(iter.product(range(1,7),repeat=num_of_dice))
    dice_state_probabilities = {}
    for d_state in all_possible_dice_states:
        index = ''.join(str(x) for x in np.sort(d_state))
        if index not in dice_state_probabilities:
            dice_state_probabilities[index] = 0
        dice_state_probabilities[index] += 1
    for d in dice_state_probabilities:
        dice_state_probabilities[d] = dice_state_probabilities[d]/len(all_possible_dice_states)
    return dice_state_probabilities

class MCTS:
    def __init__(self, game, args):
        self.game = game
        self.args = args
    
    #returns the probabilities for the possible actions
    def search(self,state,player,action,chance,dice_node,rethrow,dices):
        try:
            l = len(dices)
        except:
            l = 0
        root = Node(self.game,self.args,state,player,None,action,chance,dice_node,rethrow,l,dices)
        var = tqdm(range(self.args['num_searches']))

        for search in var:
            node = root
            while node.is_fully_expanded():
                node = node.select()
                
            points, is_terminal = self.game.is_terminated(node.state)
            value = np.argmax(points) if np.count_nonzero(points==np.max(points))==1 else -1

            if not is_terminal:
                node = node.expand()
                value = node.simulate()
            node.backpropagate(value)

        action_probs = np.zeros(3) if dice_node else np.zeros(2) if rethrow else np.zeros(20) 

        for child_key, child_value in root.children.items():
            action_probs[child_key] += child_value.visit_count

        action_probs /= np.sum(action_probs)
        depth = self.calc_depth(root)
        print("depth: ", depth)
        return action_probs

    def calc_depth(self,root):
        maxi = 0	
        if len(root.children) == 0:
            return 1
        for key,child in root.children.items():
            val = self.calc_depth(child)
            if  val > maxi:
                maxi = val 
        return maxi+1
    
# machikoro = Machikoro()
# state = machikoro.get_initial_state(2)
# encoded = machikoro.get_encoded_state(state)
# player = 0
# print(state)
# print(encoded)
# print(len(encoded))
#state[3][0][3] = 1
# print(state)
# print(machikoro.get_valid_moves(state,player))
# encoded = machikoro.get_encoded_state(state)
# print(encoded, "\n Length of encoded state: ", len(encoded))
# for i in range(1):

#     args = {
#         'C': 3,
#         'num_searches': 300
#     }
#     print("C:",args['C'])
#     mcts = MCTS(machikoro, args)

#     if player == 0:
#         mcts_probs = mcts.search(state,player,None,False,False,False,0)
#         print(mcts_probs)
#         action = np.argmax(mcts_probs)
#         print(action)
#         print(np.argsort(mcts_probs))

#Wenn die KI keine Option hat soll keine MCTS Suche gemacht werden Bsp. keine diceNode suche wenn das erste Upgrade nicht verfügbar ist
