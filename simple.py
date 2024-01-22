import numpy as np
from numpy import random
import random as r
import time
import itertools as iter
import math
import copy
from tqdm import tqdm
#Focus first only on 1v1s
def rethrow(dice, to_rethrow):
    assert(len(dice) == len(to_rethrow))
    for i in range(len(dice)):
        if to_rethrow[i]:
            dice[i] = random.randint(1,6) 
    return np.sort(dice)

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

class Yahtzee:
    def __init__(self,num_players):
        assert num_players in [1,2,3,4]
        self.player_num = num_players

    def get_valid_moves(self,state,player,rethrow):
        return np.ravel(np.argwhere(np.append(rethrow<2,state[1][player] == -1)))

    def get_next_state(self,state,player,action,re_dice=None):
        if re_dice == None:
            re_dice = np.zeros(len(state[0]))
        if action == 0:
            state[0] = rethrow(state[0],re_dice)
        else:
            state[1][player][action-1] = options[action-1](state[0])
        return state

    def get_initial_state(self,num_of_dice=5):
        state = np.array([
            np.zeros(num_of_dice),
            np.ones((self.player_num,len(options))) * -1
        ],dtype=object)
        return state

    def get_points_and_terminated(self,state):
        return np.sum(state[1],axis=1), not np.any(state[1]==-1)#np.sum(state[1]>=0,axis=1), not np.any(state[1]==-1)

class Node:
    #initialises a node with game, gamestate, the latest active player, parent node, last action taken, wether the node is chance or descision and rethrow choice which is only relevant for last action 0
    def __init__(self, game, args, state, active_player, parent=None, action_taken=None, ischance = False, rethrow_choice =  None, throw=0):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.ischance = ischance
        self.children = {}
        self.active_player = active_player
        self.throw = throw
        #one list of moves/or rethrow choices for descision nodes| a list of possible dice states plus their probobilities
        self.rethrow_choice = rethrow_choice
        self.expandable_moves = calc_sorted_possible_dice_states(self.state[0],self.rethrow_choice) if ischance else copy.deepcopy(all_permutations) if action_taken==0 else game.get_valid_moves(self.state,self.active_player,self.throw)
            
        self.visit_count = 0
        self.value_sum = 0

    def is_fully_expanded(self):
        if self.ischance:
            return len(self.expandable_moves[0]) == len(self.children) and len(self.children) > 0
        return len(self.expandable_moves) == len(self.children) and len(self.children) > 0
    
    def select(self):
        if self.ischance:#for chance nodes expandle_moves[1] contains the probability distribution 
            dsp = self.expandable_moves[1]
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

    # def get_ucb(self,child):########################################
    #     q_value = 1- ((child.value_sum / child.visit_count) + 1) / 2
    #     return q_value + self.args['C'] * math.sqrt(math.log(self.visit_count) / child.visit_count)

    def get_ucb(self,child):
        #q_value is normalised from [-1,1] 
        if self.active_player==child.active_player:
            q_value = ((child.value_sum / child.visit_count) + 1) / 2
        else:#if the child has a different active player the q value is inverted because the score is from a different view
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.args['C'] * math.sqrt(math.log(self.visit_count) / child.visit_count)  
     
    def expand(self):
        if self.action_taken == 0:
            index = r.choice(np.ravel(np.argwhere([self.expandable_moves[i]!=-1 for i in range(len(self.expandable_moves))])))
            permutation = self.expandable_moves[index]
            child_state = copy.deepcopy(self.state)
            child = Node(self.game,self.args,child_state,self.active_player,self,None,True,permutation,self.throw+1)#add node with specific rethrow choice
            child.expand()###
            self.children[''.join(str(x) for x in permutation)] = child
            self.expandable_moves[index] = -1

        #append all chance outcomes following node is decision node
        elif self.ischance:
            for dices in self.expandable_moves[0]:
                child_state = copy.deepcopy(self.state)
                child_state[0] = np.asarray(dices)
                child = Node(self.game,self.args,child_state,self.active_player,self,None,False,throw=self.throw)#add node for all dice outcomes
                self.children[''.join(str(x) for x in dices)] = child

        #normal decision node case following node is either chance node or decision node for rethrow
        else:
            assert self.parent == None or self.parent.ischance
            action = self.expandable_moves[r.choice(np.where(self.expandable_moves!=-1)[0])]
            child_state = copy.deepcopy(self.state)
            if action == 0:
                
                child = Node(self.game,self.args,child_state,self.active_player,self,action,False,throw=self.throw)#add rethrow node where rethrow choice is TBD
            else:
                child_state = self.game.get_next_state(child_state,self.active_player,action)
                #the child with the new player doesn't have new dice since this is handled in the children of the child since the child is chance
                child = Node(self.game,self.args,child_state,(self.active_player+1)%len(child_state[1]),self,action,True)#add node for new players turn
                assert self.active_player != child.active_player
                child.expand()###

            self.children[action] = child
            self.expandable_moves[np.argwhere(self.expandable_moves==action)[0][0]]  = -1

        return child
    
    def simulate(self):
        assert self.parent is not None #a simulation is made onto a child thus the simulation node should have a parent
        points, is_terminal = self.game.get_points_and_terminated(self.state)
        value = np.argmax(points) if np.count_nonzero(points==np.max(points))==1 else -1
        if is_terminal:
            return value
        
        rollout_state = copy.deepcopy(self.state)
        player = self.active_player
        throw = self.throw
        #simulate with specific rethrow choice
        if self.ischance and self.action_taken is None:
            #throw = self.throw
            assert throw > 0 
            rollout_state = self.game.get_next_state(rollout_state,player,0,self.rethrow_choice)
            v = self.game.get_valid_moves(rollout_state,player,throw)
            act = r.choice(v)
            while act == 0 and throw<2:
                rollout_state = self.game.get_next_state(rollout_state,player,act,r.choice(all_permutations))
                v = self.game.get_valid_moves(rollout_state,player,throw)
                throw += 1
                act = r.choice(v)
            rollout_state = self.game.get_next_state(rollout_state,player,act)
            rollout_state = self.game.get_next_state(rollout_state,player,0,[1,1,1])
            player = (player+1)%len(self.state[1])
            throw = 0
            points, is_terminal = self.game.get_points_and_terminated(rollout_state)
        #simulate after rethrow was chosen but specific rethrow choice (-> pick any rethrow)
        elif not self.ischance and self.action_taken==0:
            rollout_state = self.game.get_next_state(rollout_state,player,0,r.choice(all_permutations)) 
            throw += 1
        #action was chosen so the game must simulate the node with new player and use new dice
        elif self.ischance and self.action_taken > 0:
            rollout_state = self.game.get_next_state(rollout_state,player,0,(1,1,1)) 
        else:
            print("Node data: self.ischance = {}, self.action_taken = {}".format(self.ischance,self.action_taken))
            raise Exception("Else Case shouldn't be reachable due to Expansion implementation")
        while not is_terminal:
            v = self.game.get_valid_moves(rollout_state,player,throw)
            act = r.choice(v)
            while act == 0 and throw<2:
                rollout_state = self.game.get_next_state(rollout_state,player,act,r.choice(all_permutations))
                v = self.game.get_valid_moves(rollout_state,player,throw)
                throw += 1
                act = r.choice(v)

            rollout_state = self.game.get_next_state(rollout_state,player,act)
            rollout_state = self.game.get_next_state(rollout_state,player,0,[1,1,1])
            player = (player+1) %len(self.state[1])
            throw = 0
            points, is_terminal = self.game.get_points_and_terminated(rollout_state)

        return np.argmax(points) if np.count_nonzero(points==np.max(points))==1 else -1
    
    def backpropagate(self,value):
        #2-player simulation returns either 0 or 1 for the corresponding winner
        #a tie returns -1 meaning no winner; MCTS shouldn't strive for a tie #test 0 for tie
        self.value_sum += (-1)**(value!=self.active_player) * (value>=0) #* (1+ (np.sum(self.state[1][self.active_player])/np.sum(self.state[1]))) #statt self.state, das ergebnis der simulation/ die Punkte der simulation
        self.visit_count += 1
         
        if self.parent is not None:
            self.parent.backpropagate(value) 


class MCTS:
    def __init__(self, game, args):
        self.game = game
        self.args = args
    
    #returns the probabilities for the possible actions
    def search(self,state,player,last_action=None,throw=0):
        root = Node(self.game,self.args,state,player,action_taken=last_action,throw=throw)
        for search in tqdm(range(self.args['num_searches'])):
            node = root
            
            while node.is_fully_expanded():
                node = node.select()
            
            points, is_terminal = self.game.get_points_and_terminated(node.state)
            value = np.argmax(points) if np.count_nonzero(points==np.max(points))==1 else -1

            if not is_terminal:
                node = node.expand()
                value = node.simulate()
            
            node.backpropagate(value)

        action_probs = np.zeros(len(root.children)) if root.action_taken==0 else np.zeros(len(options)+1)
        x = np.zeros(len(root.children)) if root.action_taken==0 else np.zeros(len(options)+1)
        for child_key, child_value in root.children.items():

            try:#child key is integer
                action_probs[child_key] += child_value.visit_count
                x[child_key] += root.get_ucb(child_value)
            except:
                index = np.argwhere([perm == tuple([int(i) for a,i in enumerate(child_key)]) for perm in all_permutations])
                action_probs[index] += child_value.visit_count
                x[index] += root.get_ucb(child_value)
        #print(action_probs)
        #print(x)
        print(np.argmax(x))
        action_probs /= np.sum(action_probs)
        #print(self.calc_depth(root))
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

possible_dice_states = list(iter.product(range(1,7),repeat=3)) #len =  216
sorted_possible_dice_states = list(iter.combinations_with_replacement(range(1,7),r=3)) #len = 56
all_permutations = list(iter.product(range(0,2),repeat=3))[1:] #rethrow option (0,0,0) is filtered out
#calculates the probability distribution of a given number of dice_states; dice_states are handled in their sorted form
def calc_dice_state_probabilities(all_possible_dice_states):
    dice_state_probabilities = {}
    for d_state in all_possible_dice_states:
        sorted_d_state = np.sort(d_state)
        index = ''.join(str(x) for x in np.sort(sorted_d_state))
        if index not in dice_state_probabilities:
            dice_state_probabilities[index] = 0
        dice_state_probabilities[index] += 1
    for d in dice_state_probabilities:
        dice_state_probabilities[d] = dice_state_probabilities[d]/len(all_possible_dice_states)
    return dice_state_probabilities

#returns a list of poossible dice states if a specific number of dices are changed only
def calc_sorted_possible_dice_states(current_dice, changing_dice):
    if changing_dice is None or np.all(changing_dice):
        return sorted_possible_dice_states, calc_dice_state_probabilities(possible_dice_states)

    sorted_states = list()
    state_list = list()
    for elem in possible_dice_states:
        is_similar = True
        for i in range(len(current_dice)):
            # if not changing_dice[i] and elem.count(current_dice[i])<np.count_nonzero(np.ma.masked_array(current_dice,mask=changing_dice)==current_dice[i]):
            #     is_similar = False
            #     break

            if changing_dice[i] == 0 and  current_dice[i] != elem[i]:
                is_similar = False
                break

        if is_similar:
            state_list.append(elem)
            if tuple(sorted(elem)) not in sorted_states:
                sorted_states.append(tuple(sorted(elem))) 

    return sorted_states,calc_dice_state_probabilities(state_list)

# yahtzee = Yahtzee(2)
# state = yahtzee.get_initial_state()
# player = 0
# node = Node(yahtzee,{'C':1.41},state,player,action_taken=1)
# node = node.expand()

def random_bot_action(game,state,player,valid_actions):
    c = r.choice(valid_actions)
    throw = 0
    while c==0 and throw<2:
        state = game.get_next_state(state,player,0,np.asarray(r.choice(all_permutations)))
        v = game.get_valid_moves(state,player,throw)
        c = r.choice(v)
        throw += 1
    return c

def classic_greedy_bot(game,state,player,valid_actions):
    option = valid_actions[1:]
    choice,points = -1,0
    t = 0
    while t<2:
        for opt in range(len(option)):       
            c_game = copy.deepcopy(state)
            c_game = game.get_next_state(c_game,player,option[opt])
            p = options[option[opt]-1](c_game[0])
            if p>points:
                choice,points = option[opt],p
        if points == 0:
            state = game.get_next_state(state,player,0,[1,1,1])
            t+=1
        else:
            break
    if choice == -1:
        choice = r.choice(option)
    return choice

yahtzee = Yahtzee(2)
player = 0
state = yahtzee.get_initial_state(3)
state[0] = np.array([1,1,2])


# state[1][0][0] = 0
# state[1][1][0] = 0
# state[1][0][1] = 0
# state[1][1][1] = 0
# state[1][0][2] = 0
# state[1][1][2] = 0
print(all_permutations)

print(state)
for i in range(1):
    
    args = {
        'C': 1.41,
        'num_searches': 10000
    }
    print("C:",args['C'])
    mcts = MCTS(yahtzee, args)
  
    if player == 0:
        mcts_probs = mcts.search(state,player)
        print(mcts_probs)
        action = np.argmax(mcts_probs)
        print(action)
        print(np.argsort(mcts_probs))
    else:
        action = random_bot_action(yahtzee,state,player,yahtzee.get_valid_moves(state,player,0))
    
# player = 0
# throw = 0
# action = -1
# yahtzee = Yahtzee(2)
# state = yahtzee.get_initial_state(3)
# state = yahtzee.get_next_state(state,player,0,(1,1,1))
# args = {
#     'C': 3.5,
#     'num_searches': 10000
# }
# print("C:",args['C'])
# mcts = MCTS(yahtzee, args)
# points, is_terminated = yahtzee.get_points_and_terminated(state)

# print(state)
# while not is_terminated:
#     print("ACTIVE PLAYER IS PLAYER ",player)
#     mcts_probs = np.array([])
#     if player == 0:
#         mcts_probs = mcts.search(state,player,action,throw)
#         action = np.argmax(mcts_probs)           
#     else:
#         action = classic_greedy_bot(yahtzee,state,player,yahtzee.get_valid_moves(state,player,throw))

#     if len(mcts_probs)==(len(options)+1) and action == 0: #rand bot will never return 0 since rethrow is internally
#         continue
#     elif len(mcts_probs)==len(all_permutations):
#         print("dice before choice:\n",state[0])
#         state = yahtzee.get_next_state(state,player,0,all_permutations[action])
#         throw += 1
#         print("rethrow choice: ", all_permutations[action])
#         print("state after choice:\n",state)
#         continue
#     else:
#         state = yahtzee.get_next_state(state,player,action)
#         print("Throw: ",throw)
#         print("action taken: ",action)
#         print("state after action:\n",state)
#         state = yahtzee.get_next_state(state,player,0,(1,1,1))
#         player = (player+1)%len(state[1])        
#         throw = 0
#     print("\n")
#     points, is_terminated = yahtzee.get_points_and_terminated(state)
# print(state)
# print(points)
        