import numpy as np
from numpy import random
import itertools as iter
import random as r
import itertools as iter
import copy
from tqdm import tqdm
import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import machikoro

print(torch.__version__)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

class NeuralNetwork(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.device = device 

        self.policyHead = nn.Sequential(
            nn.Linear(65, 44,dtype=float),
            nn.ReLU(),
            nn.Linear(44, 23,dtype=float)
        )

        self.valueHead = nn.Sequential(
            nn.Linear(65, 33,dtype=float),
            nn.ReLU(),
            nn.Linear(33, 1,dtype=float),
            nn.Tanh()
        )

        self.to(device)

    def forward(self, x):
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return value, policy
    
class Node:

    def __init__(self, game, args, state, active_player, parent=None, action_taken=None, ischance = False, dices = None, has_used_rethrow = False, prior=0):
        self.game = game
        self.state = state
        self.args = args
        self.active_player = active_player
        self.parent = parent
        self.children = {}

        self.action_taken = action_taken
        self.ischance = ischance
        self.dices = dices
        self.has_used_rethrow = has_used_rethrow
        self.prior = prior

        self.expandable_moves = calc_dice_state_probabilities(len(dices)) if ischance else None
        self.visit_count = 0
        self.value_sum = 0

    def is_fully_expanded(self):
        return len(self.children) > 0
    
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
            for dices in self.expandable_moves.keys():
                child_state = copy.deepcopy(self.state)
                can_rethrow = self.state[2][self.active_player][3] and (self.parent is None or self.parent.isrethrow_node == False)###
                if not can_rethrow or self.has_used_rethrow:
                    self.game.distribution(child_state,self.active_player,np.array([int(x) for x in dices]))
                child = Node(self.game,self.args,child_state,self.active_player,self,None,False,dices)
                self.children[dices] = child
        else:
            for action, prob in enumerate(policy):
                if prob > 0:
                    child_state = copy.deepcopy(self.state)
                    if action <= 19:
                        child_state = self.game.get_next_state(child_state,self.active_player,action)
                        child = Node(self.game,self.args,child_state,(self.active_player+1) % len(child_state[0]), self, action, False, None, False, prob)
                    elif action == 22:
                        self.game.distribution(child_state,self.active_player,np.array([int(x) for x in self.dices]))
                        child = Node(self.game,self.args,child_state,self.active_player,self,action,False,self.dices,False,prob)
                    else:#action in (20,21,23)
                        child = Node(self.game, self.args, child_state, self.active_player, self, action, True, self.dices, action==23, prob)
                        child.expand(None)
                    self.children[action] = child
      
    # def backpropagate(self,value):
    #     self.value_sum += value 
    #     self.visit_count += 1
        
    #     if self.parent is not None:
    #         if self.active_player != self.parent.active_player:
    #             value = -value
    #         self.parent.backpropagate(value) 
    def backpropagate(self,value):
        node = self
        while not node.parent is None:
            node.value_sum += value 
            node.visit_count += 1
            
            if node.parent is not None:
                if node.active_player != node.parent.active_player:
                    value = -value
                node = node.parent

        node.value_sum += value 
        node.visit_count += 1


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
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model
    
    #returns the probabilities for the possible actions
    def search(self,state,player,last_action,dices=None):
        root = Node(self.game,self.args,state,player,None,last_action,False,dices)

        for search in range(self.args['num_searches']):
            node = root

            while node.is_fully_expanded():
                node = node.select()

            winner, is_terminal = self.game.is_terminated(node.state)
            value = -1**(winner)

            if not is_terminal:
                value, policy = self.model(torch.tensor(self.game.get_encoded_state(node.state),device=self.model.device))
                policy = torch.softmax(policy,0).detach().cpu().numpy()

                if node.action_taken is not None and node.action_taken <= 19 and node.state[2][node.active_player][0]:
                    policy[:20] = 0
                    policy[22:] = 0
                elif node.action_taken is not None and node.action_taken in [20,21] and node.state[2][node.active_player][3]:
                    policy[:22] = 0                  
                else:
                    valid_moves = self.game.get_valid_moves(node.state,node.active_player)
                    for i in range(len(policy)):
                        if i not in valid_moves:
                            policy[i] = 0

                policy /= np.sum(policy)

                value = value.item()
                node.expand(policy)
            
            node.backpropagate(value)

        action_probs = np.zeros(23)
        for child_key, child_value in root.children.items():
            action_probs[child_key] = child_value.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs

class AlphaZero:
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTS(game,args,model)

    def selfPlay(self):
        memory = []
        player = 0
        state = self.game.get_initial_state(2)

        action = None
        dices = None
        dice_choice_available = False
        rethrow_choice_available = False
        zero_cnt = 0
        while True:
            if dice_choice_available:
                action_probs = self.mcts.search(state,player,action,dices)###
                
                action = r.choices(np.arange(len(action_probs)),action_probs)[0]-19
                dices = machikoro.dice(action)

                memory.append((state, action_probs, player))

                dice_choice_available = False
            elif rethrow_choice_available:
                action_probs = self.mcts.search(state,player,action,dices)###

                action = r.choices(np.arange(len(action_probs)),action_probs)[0] - 22

                dices = machikoro.dice(len(dices)) if action else dices

                memory.append((state, action_probs, player))

                rethrow_choice_available = False
            else:

                action_probs = self.mcts.search(state,player,action,dices)

                memory.append((state, action_probs, player))

                action = r.choices(np.arange(len(action_probs)),action_probs)[0]

                state = self.game.get_next_state(state,player,action)

                winner, is_terminal = self.game.is_terminated(state)

                if is_terminal or zero_cnt == 20:
                    returnMemory = []
                    for hist_neutral_state, hist_action_probs, hist_player in memory:
                        hist_outcome = 1 if hist_player == winner else -1
                        returnMemory.append((
                            self.game.get_encoded_state(hist_neutral_state),
                            hist_action_probs,
                            hist_outcome
                        ))
                    return returnMemory
                
                if action == 0:
                    zero_cnt += 1
                else:
                    zero_cnt = 0
                
                player = (player +1) % len(state[0])
                dice_choice_available = state[2][player][0]
                rethrow_choice_available = state[2][player][3]

    def train(self,memory):
        random.shuffle(memory)
        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIdx:min(len(memory) - 1, batchIdx + self.args['batch_size'])] # Change to memory[batchIdx:batchIdx+self.args['batch_size']] in case of an error
            state, policy_targets, value_targets = zip(*sample)
            
            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(value_targets).reshape(-1, 1)
            
            state = torch.tensor(state, dtype=float, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=float, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=float, device=self.model.device)
            
            out_value, out_policy = self.model(state)
            #print(out_policy.shape,"\n",policy_targets.shape,"\n",out_value.shape,"\n",value_targets.shape,"\n")
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
            for selfPlay_iteration in tqdm(range(self.args['num_selfPlay_iterations'])):
                memory += self.selfPlay()
                
            self.model.train()
            for epoch in tqdm(range(self.args['num_epochs'])):
                self.train(memory)
            
            torch.save(self.model.state_dict(), f"model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"optimizer_{iteration}.pt")


def test():
    mk = machikoro.Machikoro()
    model = NeuralNetwork(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    args = {
        'C': 2,
        'num_searches': 100,
        'num_iterations': 2,
        'num_selfPlay_iterations': 1,
        'num_epochs': 4,
        'batch_size': 64
    }

    alphaZero = AlphaZero(model, optimizer, mk, args)
    alphaZero.learn()

test()