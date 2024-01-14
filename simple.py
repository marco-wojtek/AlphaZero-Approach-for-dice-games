import numpy as np
from numpy import random
import random as r
import time
import itertools as iter
import math
import copy
#Focus first only on 1v1s
def rethrow(dice, to_rethrow):
    assert(len(dice) == len(to_rethrow))
    for i in range(len(dice)):
        if to_rethrow[i]:
            dice[i] = random.randint(1,6) 
    return dice

def straight(dice):
    cnt_longest_seq = 1
    for n in range(1,len(dice)):
        if dice[n] == dice[n-1]+1:
            cnt_longest_seq += 1
    return True if cnt_longest_seq == 3 else False 

options = np.array([
           lambda x: 20 if straight(x) else 0, #straight
           lambda x: 30 if np.count_nonzero(x==x[0]) == 3 else 0,#yahtzee
           lambda x: np.sum(x)#chance
           ])

class yahtzee:
    def __init__(self,num_players):
        assert num_players in [1,2,3,4]
        self.player_num = num_players

    def get_valid_moves(self,state,player,rethrow):
        return np.ravel(np.argwhere(np.append(rethrow<2,state[1][player] == -1)))

    def get_next_state(self,state,player,action,re_dice=[0,0,0,0,0]):
        if action == 0:
            state[0] = rethrow(state[0],re_dice)
        else:
            state[1][player][action-1] = options[action-1](state[0])
        return state

    def get_initial_state(self):
        state = np.array([
            np.zeros(3),
            np.ones((self.player_num,len(options))) * -1
        ],dtype=object)
        return state

    def get_points_and_terminated(self,state):
        return np.sum(state[1]>0,axis=1), not np.any(state[1]==-1)

class Node:
    def __init__(self, game, args, state, player, parent=None, action_taken=None, ischance = False):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.ischance = ischance
        self.children = {}
        self.expandable_moves = game.get_valid_moves(state)
        self.active_player = player
        
        self.visit_count = 0
        self.value_sum = 0

    def is_fully_expanded(self):
        return len(self.expandable_moves) == 0 and len(self.children) > 0
    
    def select(self):
        if self.ischance:
            outcome = r.choices(list(dice_state_probabilities.keys()),list(dice_state_probabilities.values()))
            return self.children[outcome]
        
        best_child = None
        best_ucb = -np.inf
        
        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb
                
        return best_child
    
    def get_ucb(self,child):
        q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.args['C'] * math.sqrt(math.log(self.visit_count) / child.visit_count)
    
    def expand(self):
        action = r.choice(self.expandable_moves)
        self.expandable_moves[action] = -1

        #append all possible dice outcomes to the current node -> maybe need reposition
        # if self.ischance:
        #     for dices in sorted_possible_dice_states:
        #         child_state = copy.deepcopy(self.state)
        #         child_state[0] = np.asarray(dices)
        #         self.children[''.join(str(x) for x in dices)] = Node(self.game,self.args,child_state,self.active_player,self,None,False)
        #TODO
        return
    
    def simulate(self):
        points, is_terminal = self.game.get_points_and_terminated(self.state)
        value = np.argmax(points)
        if is_terminal:
            return value
        
        rollout_state = copy.deepcopy(self.state)
        #TODO
        return
    
    def backpropagate(self,value):
        self.value_sum += value
        self.visit_count += 1
        
        # value = 
        if self.parent is not None:
            self.parent.backpropagate(value) 

possible_dice_states = list(iter.product(range(1,7),repeat=3)) #len =  216
sorted_possible_dice_states = list(iter.combinations_with_replacement(range(1,7),r=3)) #len = 56
#contains the probability distribution for sorted dice events to happen
dice_state_probabilities = {}
for d_state in possible_dice_states:
    sorted_d_state = np.sort(d_state)
    index = ''.join(str(x) for x in np.sort(sorted_d_state))
    if index not in dice_state_probabilities:
        dice_state_probabilities[index] = 0
    dice_state_probabilities[index] += 1
for d in dice_state_probabilities:
    dice_state_probabilities[d] = dice_state_probabilities[d]/len(possible_dice_states)

Yahtzee = yahtzee(2)
state = Yahtzee.get_initial_state()
child = {}
i = 0
for dices in sorted_possible_dice_states:
    child_state = copy.deepcopy(state)
    child[''.join(str(x) for x in dices)] = state
    i +=1
    if i==3:
        break

print(dice_state_probabilities)
outcome = r.choices(list(dice_state_probabilities.keys()),weights=list(dice_state_probabilities.values()))
print(outcome)

