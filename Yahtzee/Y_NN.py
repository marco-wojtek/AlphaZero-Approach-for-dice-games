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
# from torch.utils.data import DataLoader
# from torchvision import datasets
# from torchvision.transforms import ToTensor

import y
print(torch.__version__)

#run with python -u "d:\Informatikstudium\Bachelor-Arbeit\Python_code\NN.py" or pyton -u NN.py
device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.device = device 
        #sqrt(input layer nodes * output layer nodes)
        self.policyHead = nn.Sequential(
            # nn.Linear(77, 128,dtype=float),
            # nn.ReLU(),
            # nn.Linear(128, 128,dtype=float),
            # nn.ReLU(),
            # nn.Linear(128, 44,dtype=float)
            nn.Linear(64, 128,dtype=float),
            nn.ReLU(),
            nn.Linear(128, 64,dtype=float),
            nn.ReLU(),
            nn.Linear(64, 44,dtype=float)
        )

        self.valueHead = nn.Sequential(
            # nn.Linear(74, 64,dtype=float),
            # nn.ReLU(),
            # nn.Linear(64, 1,dtype=float),
            # nn.Tanh()
            # nn.Linear(77, 64,dtype=float),
            # nn.ReLU(),
            # nn.Linear(64, 64,dtype=float),
            # nn.ReLU(),
            # nn.Linear(64, 1,dtype=float),
            # nn.Tanh()
            nn.Linear(64, 64,dtype=float),
            nn.ReLU(),
            nn.Linear(64, 32,dtype=float),
            nn.ReLU(),
            nn.Linear(32, 1,dtype=float),
            nn.Tanh()
        )

        self.to(device)

    def forward(self, x):
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return value, policy
    
all_permutations = list(iter.product(range(0,2),repeat=5))[1:]#31
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
    def __init__(self, game, args, state, active_player, parent=None, action_taken=None, ischance = False, rethrow_choice =  None, throw=0, prior = 0,visit_count=0):
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
            
        self.visit_count = visit_count
        self.value_sum = 0

    def is_fully_expanded(self):
        return len(self.children) > 0
    
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
                    if action <=12:#action is choosing to enter a throw
                        child_state = self.game.get_next_state(child_state,self.active_player,action)
                        child = Node(self.game, self.args, child_state, (self.active_player+1)%len(child_state[1]),self, action, True, None, 0, prob)
                        child.expand(None)
                    else:#action is choosing a rethrow constellation
                        child = Node(self.game, self.args, child_state, self.active_player, self, None, True, all_permutations[action-13], self.throw+1, prob)
                        child.expand(None)
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
        root = Node(self.game,self.args,state,player,action_taken=last_action,throw=throw,visit_count=1)
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
                valid_moves = self.game.get_valid_moves(node.state,node.active_player,node.throw)
                for i in range(len(policy)):
                    if i not in valid_moves:
                        policy[i] = 0
                #policy = torch.softmax(policy,0).detach().cpu().numpy()
                policy /= np.sum(policy)

                value = value.item()
                node.expand(policy)
            node.backpropagate(value)
        action_probs = np.zeros(44)
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
        state = self.game.get_initial_state()
        state = self.game.get_next_state(state,player,-1,(1,1,1,1,1))

        action = None
        throw = 0
        while True:
            #print(state, "\n action: ",action , "\n -------------------------")
            action_probs = self.mcts.search(state,player,action,throw)#check search in relation to valid moves
            #print("action_props: ", action_probs)
            memory.append((state,action_probs,player))

            temperature_action_probs = action_probs ** (1 / self.args['temperature'])#squishes the values together to allow more exploration
            action = r.choices(np.arange(len(action_probs)),temperature_action_probs)[0]
            #print("-------------------------------")
            if action <=12:
                state = self.game.get_next_state(state,player,action)
            else:
                state = self.game.get_next_state(state,player,-1,all_permutations[action-13])

            #check points and terminated
            value, is_terminal = self.game.get_points_and_terminated(state)

            if is_terminal:
                returnMemory = []
                for hist_neutral_state, hist_action_probs, hist_player in memory:
                    hist_outcome = 1 if hist_player == np.argmax(value) else -1
                    returnMemory.append((
                        self.game.get_encoded_state(hist_neutral_state),
                        hist_action_probs,
                        hist_outcome
                    ))
                return returnMemory

            #change turn
            if action<=12:
                player = (player+1)%len(state[1])
                state = self.game.get_next_state(state,player,0,(1,1,1,1,1))
                throw = 0
            else:
                throw += 1


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
            assert torch.all(value_targets>=-1 | value_targets<=1).item() ###delete tanh
            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss
            print(policy_loss)
            print(value_loss)
            print(loss)
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
            
            torch.save(self.model.state_dict(), f"Models/model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"Models/optimizer_{iteration}.pt")



def test():
    yahtzee = y.Yahtzee(2)
    model = NeuralNetwork(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    args = {
        'C': 2,
        'num_searches': 300,
        'num_iterations': 1,
        'num_selfPlay_iterations': 3,
        "num_parallel_games" : 100,
        'num_epochs': 4,
        'batch_size': 64,
        'temperature': 1.25
    }

    alphaZero = AlphaZero(model, optimizer, yahtzee, args)
    alphaZero.learn()

#test()
    

# args = {
#     'C': 2,
#     'num_searches': 300,
# }

# model 0
# [43.1215 49.3035]
# [131. 165.]
# [ 9. 12.]
# [38. 45.]

# model B 0 
# [43.1077 47.121 ]
# [147. 157.]
# [6. 8.]
# [38. 43.]
    
# yahtzee = y.Yahtzee(2)
# state = yahtzee.get_initial_state()
# state = yahtzee.get_next_state(state,0,0,(1,1,1,1,1))

# model = NeuralNetwork(device)
# model.load_state_dict(torch.load('Models/model_0.pt', map_location=device))


# value, policy = model(torch.tensor(yahtzee.get_encoded_state(state),device=model.device))
# policy = torch.softmax(policy,0).detach().cpu().numpy()
# # print(policy)
# # policy[14:] = 0
# # policy /= np.sum(policy)
# print(state)
# print(policy)
# print(np.argmax(policy))
# print(np.argsort(policy))


# value, policy = model(torch.tensor(yahtzee.get_encoded_state(state),device=model.device))
# policy = torch.softmax(policy,0).detach().cpu().numpy()
# policy[14:] = 0
# policy /= np.sum(policy)
# print(policy)
# print(np.argmax(policy))

class MCTSParallel:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model
    
    @torch.no_grad()
    def search(self,states,player,spGames,last_action=None,throw=0):
        #dirichlet variant
        _, policy = self.model(torch.tensor(self.game.get_encoded_states(states,throw),device=self.model.device))
        policy = torch.softmax(policy,1).detach().cpu().numpy()
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] * np.random.dirichlet([self.args['dirichlet_alpha']] * len(policy[0]), size=policy.shape[0])
        for i,spg in enumerate(spGames):
            spg.root = Node(self.game,self.args,states[i],player[i],action_taken=last_action[i],throw=throw[i],visit_count=1)
            spg_policy = policy[i]

            node = spg.root
            valid_moves = self.game.get_valid_moves(node.state,node.active_player,node.throw)
            for k in range(len(spg_policy)):
                if k not in valid_moves:
                    spg_policy[k] = 0

            spg_policy /= np.sum(spg_policy)

            spg.root.expand(spg_policy)

        # for i,spg in enumerate(spGames):
        #     spg.root = Node(self.game,self.args,states[i],player[i],action_taken=last_action[i],throw=throw[i],visit_count=1)


        for search in range(self.args['num_searches']):
            for i,spg in enumerate(spGames):
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
                throws = [spGames[mappingIdx].node.throw for mappingIdx in expandable_spGames]
                value, policy = self.model(torch.tensor(self.game.get_encoded_states(states,throws),device=self.model.device))
                policy = torch.softmax(policy,1).detach().cpu().numpy()
                value = value.cpu().numpy()
    
            for i, mappingIdx in enumerate(expandable_spGames):
                node = spGames[mappingIdx].node
                spg_policy, spg_value = policy[i], value[i]
                valid_moves = self.game.get_valid_moves(node.state,node.active_player,node.throw)
                for k in range(len(spg_policy)):
                    if k not in valid_moves:
                        spg_policy[k] = 0
                
                # if np.sum(spg_policy) == 0:
                #     print(states)
                #     print("node state: ",node.state)
                #     print(len(spGames))
                # print(value)
                # print(spg_policy)
                # print(len(spg_policy))
                # print(np.argsort(spg_policy))
                assert np.sum(spg_policy) > 0
                # print(i)
                # print(node.state)
                    # print("active_player: ",node.active_player)
                    # print("action taken: ", node.action_taken)
                    # print("throw: ",node.throw)
                    # print(self.game.get_valid_moves(node.state,node.active_player,node.throw))
                    # print(spg_policy)
                    # print("-------------------")
                spg_policy /= np.sum(spg_policy)

                node.expand(spg_policy)
                node.backpropagate(spg_value)
        # print("last_actions: ",last_action)
        # print(states)
        # print("----------------------")
            # d = {}
            # for child_key, child_value in spg.root.children.items():
            #     d[child_key] = child_value.visit_count
            # print(d)
                
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

        #use lists so that delete function can be used
        player = list(np.zeros(len(spGames),dtype=int))
        action = len(spGames)*[None]
        throw = list(np.zeros(len(spGames),dtype=int))
        while len(spGames) > 0:
            states = np.stack([spg.state for spg in spGames])
            #print(states, "\n action: ",action , "\n -------------------------")
            self.mcts.search(states,player,spGames,action,throw)
            for i in range(len(spGames))[::-1]:
                spg = spGames[i]

                action_probs = np.zeros(44)
                for child_key, child_value in spg.root.children.items():
                    action_probs[child_key] = child_value.visit_count

                action_probs /= np.sum(action_probs)
                # print("action_props: ", action_probs)
                
                spg.memory.append((spg.root.state,action_probs,player[i]))

                temperature_action_probs = action_probs ** (1 / self.args['temperature'])#squishes the values together to allow more exploration
                action[i] = r.choices(np.arange(len(action_probs)),temperature_action_probs)[0]
                #print("-------------------------------")
                if action[i] <=12:
                    spg.state = self.game.get_next_state(spg.state,player[i],action[i])
                else:
                    spg.state = self.game.get_next_state(spg.state,player[i],-1,all_permutations[action[i]-13])

                #check points and terminated
                points, is_terminal = self.game.get_points_and_terminated(spg.state)

                if is_terminal:
                    value = 0 if np.count_nonzero(points==np.max(points)) != 1 else 1
                    for hist_neutral_state, hist_action_probs, hist_player in spg.memory:
                        hist_outcome = value if hist_player == np.argmax(points) else -value
                        return_memory.append((
                            self.game.get_encoded_state(hist_neutral_state,throw[i]),
                            hist_action_probs,
                            hist_outcome
                        ))
                    #print("------------------------------\n deleted Game is index ", i, ": ",spGames[i].state,"\n ------------------------------")
                    del spGames[i]
                    del player[i]
                    del action[i]
                    del throw[i]
                    continue

                #change turn
                if action[i]<=12:
                    player[i] = (player[i]+1)%len(spg.state[1])
                    spg.state = self.game.get_next_state(spg.state,player[i],-1,(1,1,1,1,1))
                    throw[i] = 0
                else:
                    throw[i] += 1

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

            assert torch.all(value_targets>=-1).item() and torch.all(value_targets<=1).item() ###delete tanh
            out_value, out_policy = self.model(state)
            
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
            for selfPlay_iteration in tqdm(range(self.args['num_selfPlay_iterations'] // self.args['num_parallel_games'])):
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
        self.state = game.get_next_state(self.state,0,-1,(1,1,1,1,1))
        self.memory = []
        self.root = None
        self.node = None

# def testParallel():
#     yahtzee = y.Yahtzee(2)
#     model = NeuralNetwork(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#     args = {
#         'C': 3,
#         'num_searches': 80,
#         'num_iterations': 2,
#         'num_selfPlay_iterations': 100,
#         "num_parallel_games" : 50,
#         'num_epochs': 4,
#         'batch_size': 64,
#         'temperature': 1.25
#     }

#     alphaZero = AlphaZeroParallel(model, optimizer, yahtzee, args)
#     alphaZero.learn()

# testParallel()

def testParallel():
    yahtzee = y.Yahtzee(2)
    model = NeuralNetwork(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # model.load_state_dict(torch.load('Models/model_2.pt', map_location=device))
    # optimizer.load_state_dict(torch.load(',Models/optimizer_2.pt', map_location=device)) 
    args = {
        'C': 2,
        'num_searches': 50,
        'num_iterations': 3,
        'num_selfPlay_iterations': 30,
        "num_parallel_games" : 10,
        'num_epochs': 4,
        'batch_size': 32,#64
        'temperature': 1.25,
        'dirichlet_epsilon': 0.25,
        'dirichlet_alpha': 0.2
    }

    alphaZero = AlphaZeroParallel(model, optimizer, yahtzee, args)
    alphaZero.learn()


policy_loss_arr, value_loss_arr, total_loss_arr = [], [], []
testParallel()

yahtzee = y.Yahtzee(2)
state = yahtzee.get_initial_state()
state = yahtzee.get_next_state(state,0,-1,(1,1,1,1,1))

model = NeuralNetwork(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.load_state_dict(torch.load('Models/model_0.pt', map_location=device))
optimizer.load_state_dict(torch.load('Models/optimizer_0.pt', map_location=device))

# modelB.load_state_dict(torch.load('model_0.pt', map_location=device))
# optimizerB.load_state_dict(torch.load(',optimizer_0.pt', map_location=device))

num_games = 1000
x = [[0,0]]
for i in tqdm(range(num_games)):
    yahtzee = y.Yahtzee(2)
    state = yahtzee.get_initial_state()
    state = yahtzee.get_next_state(state,0,-1,(1,1,1,1,1))
    player = 0 
    throw = 0
    points, is_terminal = yahtzee.get_points_and_terminated(state)
    while not is_terminal:
        if player == 0:
            value,policy = model(torch.tensor(yahtzee.get_encoded_state(state,throw),device=model.device))
            policy = torch.softmax(policy,0).detach().cpu().numpy()
            v = yahtzee.get_valid_moves(state,player,throw)
            for i in range(len(policy)):
                if i not in v:
                    policy[i] = 0
            
            assert np.sum(policy) > 0

            policy /= np.sum(policy)
            action = np.argmax(policy)
        else:
            v = yahtzee.get_valid_moves(state,player,throw)
            action = r.choice(v)
            # value,policy = modelB(torch.tensor(yahtzee.get_encoded_state(state,throw),device=model.device))
            # policy = torch.softmax(policy,0).detach().cpu().numpy()
            # v = yahtzee.get_valid_moves(state,player,throw)
            # for i in range(len(policy)):
            #     if i not in v:
            #         policy[i] = 0
            
            # assert np.sum(policy) > 0

            # policy /= np.sum(policy)
            # action = np.argmax(policy)
        if action > 12:
            state = yahtzee.get_next_state(state,player,-1,all_permutations[action-13])
            throw += 1
        else:
            state = yahtzee.get_next_state(state,player,action)
            state = yahtzee.get_next_state(state,player,-1,(1,1,1,1,1))
            player = (player + 1) % len(state[1])
            throw = 0
            points, is_terminal = yahtzee.get_points_and_terminated(state)
    x = np.append(x,[points],axis=0)

print(1-(np.sum(np.argmax(x[1:,:],axis=1))/num_games))
print(np.average(x[1:,:],axis=0))

