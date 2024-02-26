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

import simpleQ

print(torch.__version__)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

def dice():
    return random.randint(1,7,size=(4))

all_possible_dice_states = list(iter.product(range(1,7),repeat=4))

def calc_dice_state_probabilities(all_possible_dice_states): #turns number of all possible dice states from 1296 to 756
    dice_state_probabilities = {}
    for d_state in all_possible_dice_states:
        sorted_d_state = np.append(np.sort(d_state[:2]),d_state[2:])
        index = ''.join(str(x) for x in sorted_d_state)
        if index not in dice_state_probabilities:
            dice_state_probabilities[index] = 0
        dice_state_probabilities[index] += 1
    for d in dice_state_probabilities:
        dice_state_probabilities[d] = dice_state_probabilities[d]/len(all_possible_dice_states)
    return dice_state_probabilities

class NeuralNetwork(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.device = device 
        #sqrt(input layer nodes * output layer nodes)
        self.policyHead = nn.Sequential(
            nn.Linear(122, 52,dtype=float),
            nn.ReLU(),
            nn.Linear(52, 23,dtype=float)
        )

        self.valueHead = nn.Sequential(
            nn.Linear(122, 11,dtype=float),
            nn.ReLU(),
            nn.Linear(11, 1,dtype=float),
            nn.Tanh()
        )

        self.to(device)

    def forward(self, x):
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return value, policy

expandable_moves = calc_dice_state_probabilities(all_possible_dice_states) 
class Node:

    def __init__(self, game, args, state, active_player, parent=None, active_player_action = None, ischance = False, iswhiteturn= False, prior = 0,visit_count=0):
        self.game = game
        self.state = state
        self.args = args
        self.active_player = active_player
        self.parent = parent
        self.children = {}

        self.active_player_action = active_player_action#parses the active players action through the white dice turn
        self.ischance = ischance
        self.prior = prior
        self.iswhiteturn = iswhiteturn

        self.visit_count = visit_count
        self.value_sum = 0

    def is_fully_expanded(self):        
        return len(self.children) > 0

    def select(self):
        if self.ischance:
            dsp = expandable_moves
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
    
    def expand(self,policy):
        #expansion for chance nodes
        #expansion for white dice and coloured dice
        if self.ischance:
            for dices in expandable_moves:
                child_state = copy.deepcopy(self.state)
                child_state[0] = np.array([int(x) for x in dices])
                child = Node(self.game,self.args,child_state,(self.active_player+1)%len(self.state[-1]),self,iswhiteturn=True)
                index = ''.join(str(x) for x in dices)
                self.children[index] = child
        else:
            last_turn = (self.iswhiteturn == False)
            for action, prob in enumerate(policy):
                if prob > 0:
                    child_state = copy.deepcopy(self.state)
                    if last_turn and (action + self.active_player_action == 0):
                        act = -1
                    else:
                        act = action
                    child_state = self.game.get_next_state(child_state,self.active_player,act)
                    
                    #acton memory
                    act_mem = action if self.parent is None or self.parent.ischance else self.active_player_action

                    child = Node(self.game, self.args, child_state,(self.active_player+1)%len(child_state[1]),self, act_mem, last_turn, False, prob)
                    if last_turn:
                        child.expand(None)
                    self.children[action] = child

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


class MCTS:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model
    
    #returns the probabilities for the possible actions
    def search(self,state,player,active_player_action=None,iswhiteturn=True):
        root = Node(self.game,self.args,state,player,None,active_player_action,False,iswhiteturn,visit_count=1)

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
                value, policy = self.model(torch.tensor(self.game.get_encoded_state(node.state),device=self.model.device))
                policy = torch.softmax(policy,0).detach().cpu().numpy()

                valid_moves = self.game.get_valid_moves(node.state,node.active_player,node.iswhiteturn)
                for i in range(len(policy)):
                    if i not in valid_moves:
                        policy[i] = 0

                policy /= np.sum(policy)

                value = value.item()

                node.expand(policy)

            node.backpropagate(value)

        action_probs = np.zeros(len(np.ravel(self.game.action_space))+1)

        for child_key, child_value in root.children.items():
            action_probs[child_key] += child_value.visit_count
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
        state = self.game.get_initial_state()
        state[0] = dice()

        action = None
        act_mem = None
        white_turn = True
        temp_player = player
        while True:
            #print("player: ", temp_player, "is white turn: ", white_turn, "\n")
            #print(state,"\n action: ",action , "\n -------------------------")
            action_probs = self.mcts.search(state,player,act_mem,white_turn)

            memory.append((state, action_probs, player))

            temperature_action_probs = action_probs ** (1 / self.args['temperature'])#squishes the values together to allow more exploration
            action = r.choices(np.arange(len(action_probs)),temperature_action_probs)[0]

            if not act_mem is None and (action + act_mem==0):
                action = -1
            state = self.game.get_next_state(state,temp_player,action)

            points, is_terminal = self.game.get_points_and_terminated(state)
            value = 0 if np.count_nonzero(points==np.max(points)) != 1 else 1 
            if is_terminal:
                returnMemory = []
                for hist_neutral_state, hist_action_probs, hist_player in memory:
                    hist_outcome = value if hist_player == temp_player else -value
                    returnMemory.append((
                        self.game.get_encoded_state(hist_neutral_state),
                        hist_action_probs,
                        hist_outcome
                    ))
                return returnMemory
            
            if not white_turn:
                player = (player+1)%len(state[1])
                temp_player = player
                white_turn = True
                act_mem = None
                state[0] = dice()
            else:
                temp_player = (temp_player+1)%len(state[1])
                if temp_player == player:
                    white_turn = False
                else:
                    act_mem = action

        

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
            assert torch.all(value_targets>=-1 | value_targets<=1).item() ###delete tanh
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
    quixx = simpleQ.Quixx()
    model = NeuralNetwork(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    args = {
        'C': 2,
        'num_searches': 3000,
        'num_iterations': 1,
        'num_selfPlay_iterations': 60,
        'num_epochs': 4,
        'batch_size': 64, 
        'temperature':1.25
    }

    alphaZero = AlphaZero(model, optimizer, quixx, args)
    alphaZero.learn()

#test()

# model = NeuralNetwork(device)
# model.load_state_dict(torch.load('model_0.pt', map_location=device))
# model.eval()    
# quixx = simpleQ.Quixx()
# state = quixx.get_initial_state()
# state[0] = np.array([1,2,3,5])
# args = {
#     'C': 2,
#     'num_searches': 1000
# }
# mcts = MCTS(quixx, args, model)
# action = mcts.search(state,0)
# print(quixx.get_valid_moves(state,0,True))
# print(state)
# print(np.argmax(action))
# print(action)

class MCTSParallel:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model
    
    #returns the probabilities for the possible actions
    @torch.no_grad()
    def search(self,states,spGames,player,active_player_action,iswhiteturn):
        for i, spg in enumerate(spGames):
            spg.root = Node(self.game,self.args,states[i],player[i],None,active_player_action[i],False,iswhiteturn[i],visit_count=1)

        for search in range(self.args['num_searches']):
            for i, spg in enumerate(spGames):
                spg.node = None
                node = spg.root

                while node.is_fully_expanded():
                    node = node.select()
                    
                points, is_terminal = self.game.get_points_and_terminated(node.state)
                value = 1
                if np.count_nonzero(points==np.max(points))!=1:
                    value = 0
                elif np.argmax(points) != node.active_player:
                    value = -1

                if is_terminal:
                    node.backpropagate(value)
                else:
                    spg.node = node

            expandable_spGames = [mappingIdx for mappingIdx in range(len(spGames)) if spGames[mappingIdx].node is not None]

            if len(expandable_spGames) > 0:
                states = np.stack([spGames[mappingIdx].node.state for mappingIdx in expandable_spGames])
                value, policy = self.model(torch.tensor(self.game.get_encoded_states(states),device=self.model.device))
                policy = torch.softmax(policy,1).detach().cpu().numpy()
                value = value.detach().cpu().numpy()
            
            for i, mappingIdx in enumerate(expandable_spGames):
                node = spGames[mappingIdx].node
                spg_policy, spg_value = policy[i], value[i]
                valid_moves = self.game.get_valid_moves(node.state,node.active_player,node.iswhiteturn)

                for k in range(len(spg_policy)):
                    if k not in valid_moves:
                        spg_policy[k] = 0

                spg_policy /= np.sum(spg_policy)

                node.expand(spg_policy)

                node.backpropagate(spg_value)
    
class AlphaZeroParallel:
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTSParallel(game,args,model)

    def selfPlay(self):
        return_memory = []
        

        spGames = [SPG(self.game) for spg in range(self.args['num_parallel_games'])]
        
        player = len(spGames)*[0]
        action = None
        act_mem = len(spGames)*[None]
        white_turn = len(spGames)*[True]
        temp_player = len(spGames)*[0]
        while len(spGames) > 0:
            states = np.stack([spg.state for spg in spGames])
            #print("player: ", temp_player, "is white turn: ", white_turn, "\n")
            #print(state,"\n action: ",action , "\n -------------------------")
            # print(act_mem)
            # print(white_turn)
            # print(temp_player)
            # print(states)
            # print("---------------------")
            self.mcts.search(states,spGames,player,act_mem,white_turn)
            for i in range(len(spGames))[::-1]:
                spg = spGames[i]

                action_probs = np.zeros(len(np.ravel(self.game.action_space))+1)

                for child_key, child_value in spg.root.children.items():
                    action_probs[child_key] = child_value.visit_count
                action_probs /= np.sum(action_probs)
                spg.memory.append((spg.root.state, action_probs, player[i]))

                temperature_action_probs = action_probs ** (1 / self.args['temperature'])#squishes the values together to allow more exploration
                action = r.choices(np.arange(len(action_probs)),temperature_action_probs)[0]

                if not act_mem[i] is None and temp_player[i] == player[i] and (action + act_mem[i]==0):
                    action = -1
                spg.state = self.game.get_next_state(spg.state,temp_player[i],action)

                points, is_terminal = self.game.get_points_and_terminated(spg.state)
                
                if is_terminal:
                    value = 0 if np.count_nonzero(points==np.max(points)) != 1 else 1 
                    for hist_neutral_state, hist_action_probs, hist_player in spg.memory:
                        hist_outcome = value if hist_player == temp_player[i] else -value
                        return_memory.append((
                            self.game.get_encoded_state(hist_neutral_state),
                            hist_action_probs,
                            hist_outcome
                        ))
                    del spGames[i]
                    del act_mem[i]
                    del temp_player[i]
                    del white_turn[i]
                    del player[i]
                    continue
                    

                if not white_turn[i]:
                    player[i] = (player[i]+1)%len(spg.state[1])
                    temp_player[i] = player[i]
                    white_turn[i] = True
                    act_mem[i] = None
                    spg.state[0] = dice()
                else:
                    temp_player[i] = (temp_player[i]+1)%len(spg.state[1])
                    if temp_player[i] == player[i]:
                        white_turn[i] = False
                    else:
                        act_mem[i] = action

        return return_memory
 

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
            assert torch.all(value_targets>=-1 | value_targets<=1).item() ###delete tanh
            out_policy[policy_targets==0] = -torch.inf
            policy_loss = -torch.nan_to_num(F.log_softmax(out_policy, -1) * policy_targets).sum(-1).mean()
            #policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss
            
            policy_loss_arr.append(policy_loss.item())
            value_loss_arr.append(value_loss.item())
            total_loss_arr.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 

    def learn(self):
        for iteration in range(self.args['num_iterations']):
            memory = []
            
            self.model.eval()
            for selfPlay_iteration in tqdm(range(self.args['num_selfPlay_iterations']//self.args['num_parallel_games'])):
                memory += self.selfPlay()
                
            self.model.train()
            for epoch in tqdm(range(self.args['num_epochs'])):
                self.train(memory)
            
            torch.save(self.model.state_dict(), f"Models/model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"Models/optimizer_{iteration}.pt")
            print("avg policy loss: ", np.average(policy_loss_arr))
            print("avg value loss: ", np.average(value_loss_arr))
            print("avg total loss: ", np.average(total_loss_arr))
            policy_loss_arr.clear()
            value_loss_arr.clear()
            total_loss_arr.clear()

class SPG:
    def __init__(self,game):
        self.state = game.get_initial_state()
        self.state[0] = dice()
        self.memory = []
        self.root = None
        self.node = None

def testParallel():
    quixx = simpleQ.Quixx()
    model = NeuralNetwork(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    args = {
        'C': 3,
        'num_searches': 800,
        'num_iterations': 3,
        'num_selfPlay_iterations': 200,
        'num_parallel_games': 50,
        'num_epochs': 4,
        'batch_size': 64, 
        'temperature':1.25
    }

    alphaZero = AlphaZeroParallel(model, optimizer, quixx, args)
    alphaZero.learn()

policy_loss_arr, value_loss_arr, total_loss_arr = [], [], []
testParallel()