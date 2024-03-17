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
import matplotlib.pyplot as plt
import matplotlib as mpl
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

#hyper parameters
num_of_dice = 4
num_of_sides = 5

class NeuralNetwork(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.device = device 
        #sqrt(input layer nodes * output layer nodes)
        self.policyHead = nn.Sequential(
            nn.Linear(50, 128,dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(128, 64,dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(64, 26,dtype=torch.float32)
        )

        self.valueHead = nn.Sequential(
            nn.Linear(50, 64,dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(64, 32,dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(32, 1,dtype=torch.float32),
            nn.Tanh()
        )

        self.to(device)

    def forward(self, x):
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return value, policy
    
class NeuralNetwork2(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.device = device 
        #sqrt(input layer nodes * output layer nodes)
        self.policyHead = nn.Sequential(
            nn.Linear(50, 128,dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(128, 128,dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(128, 128,dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(128, 128,dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(128, 64,dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(64, 26,dtype=torch.float32)
        )

        self.valueHead = nn.Sequential(
            nn.Linear(50, 64,dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(64, 64,dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(64, 64,dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(64, 64,dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(64, 32,dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(32, 1,dtype=torch.float32),
            nn.Tanh()
        )

        self.to(device)

    def forward(self, x):
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return value, policy
    
class NeuralNetwork3(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.device = device 
        #sqrt(input layer nodes * output layer nodes)
        self.policyHead = nn.Sequential(
            nn.Linear(50, 128,dtype=torch.float32),
            nn.LayerNorm(128,dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(128, 64,dtype=torch.float32),
            nn.LayerNorm(64,dtype=torch.float32),
            nn.ReLU(),
            # nn.Linear(128, 64,dtype=torch.float32),
            # nn.ReLU(),
            # nn.Linear(128, 64,dtype=torch.float32),
            # nn.ReLU(),
            # nn.Linear(128, 64,dtype=torch.float32),
            # nn.ReLU(),
            nn.Linear(64, 26,dtype=torch.float32)
        )

        self.valueHead = nn.Sequential(
            nn.Linear(50, 64,dtype=torch.float32),
            nn.LayerNorm(64,dtype=torch.float32),
            nn.ReLU(),
            # nn.Linear(64, 64,dtype=torch.float32),
            # nn.ReLU(),
            # nn.Linear(64, 64,dtype=torch.float32),
            # nn.ReLU(),
            # nn.Linear(64, 64,dtype=torch.float32),
            # nn.ReLU(),
            nn.Linear(64, 32,dtype=torch.float32),
            nn.LayerNorm(32,dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(32, 1,dtype=torch.float32),
            nn.Tanh()
        )

        self.to(device)

    def forward(self, x):
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return value, policy
    
all_permutations = list(iter.product(range(0,2),repeat=num_of_dice))[1:]
all_possible_dice_states = list(iter.product(range(1,num_of_sides+1),repeat=num_of_dice))
sorted_possible_dice_states = list(iter.combinations_with_replacement(range(1,num_of_sides+1),r=num_of_dice))
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
        return len(self.children) > 0 or self.ischance
    
    def select(self):
        if self.ischance:#for chance nodes expandle_moves[1] contains the probability distribution 
            dsp = self.expandable_moves[1]
            outcome = r.choices(list(dsp.keys()),list(dsp.values()))[0]
            if not outcome in self.children:
                child_state = copy.deepcopy(self.state)
                child_state[0] = np.array(list(outcome),dtype=int)
                child = Node(self.game,self.args,child_state,self.active_player,self,None,False,None,self.throw)#add node for all dice outcomes
                self.children[outcome] = child
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
        # if self.ischance:
        #     for dices in self.expandable_moves[0]:
        #         child_state = copy.deepcopy(self.state)
        #         child_state[0] = np.asarray(dices)
        #         child = Node(self.game,self.args,child_state,self.active_player,self,None,False,None,self.throw)#add node for all dice outcomes
        #         self.children[''.join(str(x) for x in dices)] = child
        # else:
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = copy.deepcopy(self.state)
                if action <=10:#action is choosing to enter a throw
                    child_state = self.game.get_next_state(child_state,self.active_player,action)
                    child = Node(self.game, self.args, child_state, (self.active_player+1)%len(child_state[1]),self, action, True, None, 0, prob)
                else:#action is choosing a rethrow constellation
                    child = Node(self.game, self.args, child_state, self.active_player, self, None, True, all_permutations[action-len(y.options)], self.throw+1, prob)
                #child.expand(None)
                self.children[action] = child
    
    def backpropagate(self,value):###
        self.value_sum += value 
        self.visit_count += 1
         
        if self.parent is not None:
            if self.active_player != self.parent.active_player:
                value = -value
            self.parent.backpropagate(value) 

class MCTSParallel:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model
    
    @torch.no_grad()
    def search(self,states,player,spGames,last_action=None,throw=0):
        #dirichlet variant
        _, policy = self.model(torch.tensor(self.game.get_encoded_states(states,throw),device=self.model.device,dtype=torch.float32))
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
                value, policy = self.model(torch.tensor(self.game.get_encoded_states(states,throws),device=self.model.device,dtype=torch.float32))
                policy = torch.softmax(policy,1).detach().cpu().numpy()
                value = value.cpu().numpy()
    
            for i, mappingIdx in enumerate(expandable_spGames):
                node = spGames[mappingIdx].node
                spg_policy, spg_value = policy[i], value[i]
                valid_moves = self.game.get_valid_moves(node.state,node.active_player,node.throw)
                for k in range(len(spg_policy)):
                    if k not in valid_moves:
                        spg_policy[k] = 0

                assert np.sum(spg_policy) > 0

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

        #use lists so that delete function can be used
        player = list(np.zeros(len(spGames),dtype=int))
        action = len(spGames)*[None]
        throw = list(np.zeros(len(spGames),dtype=int))
        while len(spGames) > 0:
            states = np.stack([spg.state for spg in spGames])
            self.mcts.search(states,player,spGames,action,throw)
            for i in range(len(spGames))[::-1]:
                spg = spGames[i]

                action_probs = np.zeros(26)
                for child_key, child_value in spg.root.children.items():
                    action_probs[child_key] = child_value.visit_count

                action_probs /= np.sum(action_probs)
                
                spg.memory.append((spg.root.state,action_probs,player[i]))

                temperature_action_probs = action_probs ** (1 / self.args['temperature'])#squishes the values together to allow more exploration
                action[i] = r.choices(np.arange(len(action_probs)),temperature_action_probs)[0]
                if action[i] <=10:
                    spg.state = self.game.get_next_state(spg.state,player[i],action[i])
                else:
                    spg.state = self.game.get_next_state(spg.state,player[i],-1,all_permutations[action[i]-len(y.options)])

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
                if action[i]<=10:
                    player[i] = (player[i]+1)%len(spg.state[1])
                    spg.state = self.game.get_next_state(spg.state,player[i],-1,(1,1,1,1))
                    throw[i] = 0
                else:
                    throw[i] += 1

        return return_memory


    def train(self,memory):
        random.shuffle(memory)
        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIdx:batchIdx+self.args['batch_size']] # Change to memory[batchIdx:batchIdx+self.args['batch_size']] in case of an error
            state, policy_targets, value_targets = zip(*sample)
            
            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(value_targets).reshape(-1, 1)
            
            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)

            assert torch.all(value_targets>=-1).item() and torch.all(value_targets<=1).item() ###delete tanh
            out_value, out_policy = self.model(state)
            
            out_policy[policy_targets==0] = -torch.inf
            policy_loss = -torch.nan_to_num(F.log_softmax(out_policy, -1) * policy_targets).sum(-1).mean()

            #policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss
            
            if save_losses:
                policy_loss_arr.append(policy_loss.item())
                value_loss_arr.append(value_loss.item())
                total_loss_arr.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 

    def learn(self):
        for iteration in range(8,self.args['num_iterations']):
            memory = []
            print("Iteration ", iteration)
            self.model.eval()
            for selfPlay_iteration in tqdm(range(self.args['num_selfPlay_iterations'] // self.args['num_parallel_games'])):
                memory += self.selfPlay()
                
            self.model.train()
            for epoch in tqdm(range(self.args['num_epochs'])):
                self.train(memory)

                if save_losses:
                    with open(f'Losses{loss_idx}/policy_loss.txt', 'a') as f:
                        f.write('%f \n' % np.average(policy_loss_arr))
                        f.close()
                    with open(f'Losses{loss_idx}/value_loss.txt', 'a') as f:
                        f.write('%f \n' % np.average(value_loss_arr))
                        f.close()
                    with open(f'Losses{loss_idx}/total_loss.txt', 'a') as f:
                        f.write('%f \n' % np.average(total_loss_arr))
                        f.close()
                policy_loss_arr.clear()
                value_loss_arr.clear()
                total_loss_arr.clear()
            
            torch.save(self.model.state_dict(), f"Models/version_{loss_idx}_model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"Models/version_{loss_idx}_optimizer_{iteration}.pt")

            # print("avg policy loss: ", np.average(policy_loss_arr))
            # print("avg value loss: ", np.average(value_loss_arr))
            # print("avg total loss: ", np.average(total_loss_arr))
            # policy_loss_arr.clear()
            # value_loss_arr.clear()
            # total_loss_arr.clear()

class SPG:
    def __init__(self,game):
        self.state = game.get_initial_state()
        self.state = game.get_next_state(self.state,0,-1,(1,1,1,1))
        self.memory = []
        self.root = None
        self.node = None

def testParallel():
    yahtzee = y.Yahtzee(2)
    model = NeuralNetwork(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.load_state_dict(torch.load(f"Models/version_{loss_idx}_model_7.pt", map_location=device))
    optimizer.load_state_dict(torch.load(f"Models/version_{loss_idx}_optimizer_7.pt", map_location=device))

    args = {
        'C': 2.5,
        'num_searches': 250,
        'num_iterations': 16,
        'num_selfPlay_iterations': 60,
        "num_parallel_games" : 20,
        'num_epochs': 6,
        'batch_size': 64,
        'temperature': 1.3,
        'dirichlet_epsilon': 0.25,
        'dirichlet_alpha': 0.2
    }

    alphaZero = AlphaZeroParallel(model, optimizer, yahtzee, args)
    alphaZero.learn()

learning_rate = 0.001
loss_idx = int(np.log10(learning_rate**-1))
policy_loss_arr, value_loss_arr, total_loss_arr = [], [], []
save_losses = False
#delete current loss files
if save_losses:
    open(f'Losses{loss_idx}/policy_loss.txt', 'w').close()
    open(f'Losses{loss_idx}/value_loss.txt', 'w').close()
    open(f'Losses{loss_idx}/total_loss.txt', 'w').close()
#testParallel()

def simulate(num_games,P1,P2,version):
    if not P1 is None:
        P1model = NeuralNetwork(device)
        P1optimizer = torch.optim.Adam(P1model.parameters(), lr=1**-version)

        P1model.load_state_dict(torch.load(f"Models/version_{version}_model_{P1}.pt", map_location=device))
        P1optimizer.load_state_dict(torch.load(f"Models/version_{version}_optimizer_{P1}.pt", map_location=device))
        P1model.eval()
    else:
        P1model = None
        P1optimizer = None

    if not P2 is None:
        P2model = NeuralNetwork(device)
        P2optimizer = torch.optim.Adam(P2model.parameters(), lr=1**-version)

        P2model.load_state_dict(torch.load(f"Models/version_{version}_model_{P2}.pt", map_location=device))
        P2optimizer.load_state_dict(torch.load(f"Models/version_{version}_optimizer_{P2}.pt", map_location=device))
        P2model.eval()
    else:
        P2model = None
        P2optimizer = None

    x = [[0,0]]
    player_types = [P1model,P2model]
    yahtzee = y.Yahtzee(2)
    ties = 0
    for i in range(num_games):   
        state = yahtzee.get_initial_state()
        state = yahtzee.get_next_state(state,0,-1,(1,1,1,1))
        player = int(i > num_games/2)
        throw = 0
        points, is_terminal = yahtzee.get_points_and_terminated(state)
        while not is_terminal:      
            v = yahtzee.get_valid_moves(state,player,throw)
            if player_types[player] is None:
                action = r.choice(v)
            else:
                _, policy = player_types[player](torch.tensor(yahtzee.get_encoded_state(state,throw),device=player_types[player].device,dtype=torch.float32))
                policy = torch.softmax(policy,0).detach().cpu().numpy()
                
                for i in range(len(policy)):
                    if i not in v:
                        policy[i] = 0
                    
                assert np.sum(policy) > 0

                policy /= np.sum(policy)
                action = np.argmax(policy)
            if action > 10:
                state = yahtzee.get_next_state(state,player,-1,y.all_permutations[action-len(y.options)])
                throw += 1
            else:
                state = yahtzee.get_next_state(state,player,action)
                state = yahtzee.get_next_state(state,player,-1,(1,1,1,1))
                player = (player + 1) % len(state[1])
                throw = 0
                points, is_terminal = yahtzee.get_points_and_terminated(state)
        if points[0] != points[1]:
            x = np.append(x,[points],axis=0)
        else:
            ties += 1
    return 1-(np.sum(np.argmax(x[1:],axis=1))/(num_games-ties)), ties

def img_for_simulation():
    for version in range(2,5):
        winrates = []
        for i in tqdm(range(8)):
            winrates = np.append(winrates,simulate(1000,i,None,version)[0]*100)
        plt.plot(np.arange(8), winrates, label = f"lr = {10**(-version)}")

    plt.title('Yahtzee winrate against random Bot')
    plt.xlabel('Iteration')
    plt.ylabel('Win rate in %')
    plt.legend(loc="lower right")
    plt.show()

img_for_simulation()
    
#tournament
# indx = np.arange(8,dtype=int)
# np.random.shuffle(indx)
# winner = np.array([],dtype=int)
# while len(winner)!= 1:
#     winner = []
#     for i in range(0,len(indx)-1,2):
#         print(f"Gen {int(indx[i])} VS Gen {int(indx[i+1])}")
#         win, ties = simulate(1000,int(indx[i]),int(indx[i+1]),4)
#         print(win)
#         x = indx[i+1] if win < 0.5 else indx[i]
#         winner = np.append(winner,int(x))
#     indx = winner
#print(simulate(1000,3,4,2))
