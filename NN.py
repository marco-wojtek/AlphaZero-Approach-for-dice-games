import numpy as np
from numpy import random
import itertools as iter
import random as r
import itertools as iter
import copy
from tqdm import tqdm
from tqdm.notebook import trange
import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import y

print(torch.__version__)
#run with python -u "d:\Informatikstudium\Bachelor-Arbeit\Python_code\NN.py" or pyton -u NN.py
# device = (
#     "cuda"
#     if torch.cuda.is_available()
#     else "mps"
#     if torch.backends.mps.is_available()
#     else "cpu"
# )
# print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.policyHead = nn.Sequential(
            nn.Linear(74, 60,dtype=float),
            nn.ReLU(),
            nn.Linear(60, 45,dtype=float)
        )

        self.valueHead = nn.Sequential(
            nn.Linear(74, 37,dtype=float),
            nn.ReLU(),
            nn.Linear(37, 1,dtype=float),
            nn.Tanh()
        )

    def forward(self, x):
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return value, policy
    
# model = NeuralNetwork().to(device)
# print(model)

yahtzee = y.Yahtzee(2)
state = yahtzee.get_initial_state()
state[0] = y.rethrow(state[0],(1,1,1,1,1))
encoded_state = yahtzee.get_encoded_state(state)
print(state)
tensor_state = torch.tensor(encoded_state)
test = tensor_state.unsqueeze(0)

model = NeuralNetwork()
value, policy = model(tensor_state)
policy = torch.softmax(policy,0).detach().numpy()
valid_moves = yahtzee.get_valid_moves(state,0,0)
p = policy.copy()
# for action phase like this
for i in range(len(policy)):
    if i not in valid_moves:
        policy[i] = 0
# for rethrow phase like this
# policy[:15] = 0
policy /= np.sum(policy)
print("value: ", value.item())
print("policy after rescaling:\n",policy)
for action, prob in enumerate(policy):
    if prob > 0:
        print("action: {}, prob: {}".format(action,prob))
# print("---------------------------")
# p[:14] = 0
# p /= np.sum(p)
# for action, prob in enumerate(p):
#     if prob > 0:
#         print("action: {}, prob: {}".format(action,prob))
all_permutations = list(iter.product(range(0,2),repeat=5))[1:]
all_possible_dice_states = list(iter.product(range(1,7),repeat=5))#7776
sorted_possible_dice_states = list(iter.combinations_with_replacement(range(1,7),r=5))#252
def calc_dice_state_probabilities(possible_dice_states):
    dice_state_probabilities = {}
    for d_state in possible_dice_states:
        index = ''.join(str(x) for x in np.sort(d_state))
        if index not in dice_state_probabilities:
            dice_state_probabilities[index] = 0
        dice_state_probabilities[index] += 1
    for d in dice_state_probabilities:
        dice_state_probabilities[d] = dice_state_probabilities[d]/len(possible_dice_states)
    return dice_state_probabilities

def calc_sorted_possible_dice_states(current_dice, changing_dice):
    if changing_dice is None or np.all(changing_dice):
        return sorted_possible_dice_states, calc_dice_state_probabilities(all_possible_dice_states)

    sorted_states = list()
    state_list = list()
    for elem in all_possible_dice_states:
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

class Node:
    #initialises a node with game, gamestate, the latest active player, parent node, last action taken, wether the node is chance or descision and rethrow choice which is only relevant for last action 0
    def __init__(self, game, args, state, active_player, parent=None, action_taken=None, ischance = False, rethrow_choice =  None, throw=0, prior = 0):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.ischance = ischance
        self.children = {}
        self.active_player = active_player
        self.throw = throw
        self.prior = prior
        #one list of moves/or rethrow choices for descision nodes| a list of possible dice states plus their probobilities
        self.rethrow_choice = rethrow_choice
        self.expandable_moves = calc_sorted_possible_dice_states(self.state[0],self.rethrow_choice) if ischance else None
            
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
        if child.visit_count == 0:
            q_value = 0 
        elif self.active_player==child.active_player:
            q_value = ((child.value_sum / child.visit_count) + 1) / 2
        else:#if the child has a different active player the q value is inverted because the score is from a different view
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        #return q_value + self.args['C'] * math.sqrt(math.log(self.visit_count) / child.visit_count) 
        return q_value + self.args['C'] * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior  
     
    def expand(self, policy):
        if self.ischance:
            for dices in self.expandable_moves[0]:
                child_state = copy.deepcopy(self.state)
                child_state[0] = np.asarray(dices)
                child = Node(self.game,self.args,child_state,self.active_player,self,None,False,None,self.throw)#add node for all dice outcomes
                self.children[''.join(str(x) for x in dices)] = child
        else:
            for action, prob in enumerate(policy):
                if prob > 0:
                    child_state = copy.deepcopy(self.state)
                    if action <=13:#action is choosing to enter a throw or rethrow
                        child_state = self.game.get_next_state(child_state,self.active_player,action)
                        if action == 0:
                            child = Node(self.game, self.args, child_state, self.active_player, self, action, False, None, self.throw+1, prob)
                        else:
                            child = Node(self.game, self.args, child_state, (self.active_player+1)%len(child_state[1]),self, action, True, None, 0, prob)
                            child.expand()
                    else:#action is choosing a rethrow constellation
                        child = Node(self.game, self.args, child_state, self.active_player, self, None, True, all_permutations[action-14], self.throw, prob)
                        child.expand()
                    self.children[action] = child#maybe change indices for children in rethrow constellation for better overview if this version is not good
    
    def backpropagate(self,value):###
        self.value_sum += value 
        self.visit_count += 1
         
        if self.parent is not None:
            if self.active_player != self.parent.active_player:
                value = -value
            self.parent.backpropagate(value) 


class MCTS:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model
    
    @torch.no_grad()
    def search(self,state,player,last_action=None,throw=0):
        root = Node(self.game,self.args,state,player,action_taken=last_action,throw=throw)
        for search in tqdm(range(self.args['num_searches'])):
            node = root
            
            while node.is_fully_expanded():
                node = node.select()
            
            points, is_terminal = self.game.get_points_and_terminated(node.state)
            value = 1
            if np.count_nonzero(points==np.max(points))!=1:
                value = 0
            elif np.argmax(points) != node.active_player:
                value = -1

            if not is_terminal:
                value, policy = self.model(torch.tensor(self.game.get_encoded_state(node)))
                if node.action_taken == 0:
                    policy[:14] = 0
                else:
                    valid_moves = self.game.get_valid_moves(node.state,node.active_player,node.throw)
                    for i in range(len(policy)):
                        if i not in valid_moves:
                            policy[i] = 0
                policy /= np.sum(policy)

                value = value.item()
                node = node.expand(policy)
            
            node.backpropagate(value)

        action_probs = np.zeros(len(root.children)) if root.action_taken==0 else np.zeros(44)
        for child_key, child_value in root.children.items():

            try:#child key is integer
                action_probs[child_key] += child_value.visit_count
            except:
                index = np.argwhere([perm == tuple([int(i) for a,i in enumerate(child_key)]) for perm in all_permutations])
                action_probs[index] += child_value.visit_count
        print(action_probs)
        print("depth:",self.calc_depth(root))
        #self.best_child(root)
        action_probs /= np.sum(action_probs)
        return action_probs
    

class AlphaZero:
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.MCTS = MCTS(game,args,model)

    def SelfPlay():
        pass

    def train(self,memory):
        random.shuffle(memory)
        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIdx:min(len(memory) - 1, batchIdx + self.args['batch_size'])] # Change to memory[batchIdx:batchIdx+self.args['batch_size']] in case of an error
            state, policy_targets, value_targets = zip(*sample)
            
            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(value_targets).reshape(-1, 1)
            
            state = torch.tensor(state, dtype=torch.float32)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32)
            value_targets = torch.tensor(value_targets, dtype=torch.float32)
            
            out_policy, out_value = self.model(state)
            
            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 

    def learn(self):
        for iteration in range(self.args['num_iterations']):
            memory = []
            
            self.model.eval()
            for selfPlay_iteration in trange(self.args['num_selfPlay_iterations']):
                memory += self.selfPlay()
                
            self.model.train()
            for epoch in trange(self.args['num_epochs']):
                self.train(memory)
            
            torch.save(self.model.state_dict(), f"model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"optimizer_{iteration}.pt")


def test():
    yahtzee = y.Yahtzee(2)
    model = NeuralNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    args = {
        'C': 2,
        'num_searches': 60,
        'num_iterations': 3,
        'num_selfPlay_iterations': 500,
        'num_epochs': 4,
        'batch_size': 64
    }

    alphaZero = AlphaZero(model, optimizer, yahtzee, args)
    alphaZero.learn()