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
import matplotlib.pyplot as plt

import simpleQ

print(torch.__version__)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"Using {device} device")

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
            nn.Linear(122, 128,dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(128, 128,dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(128, 23,dtype=torch.float32)
        )

        self.valueHead = nn.Sequential(
            nn.Linear(122, 128,dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(128, 64,dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(64, 1,dtype=torch.float32),
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
            nn.Linear(122, 128,dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(128, 128,dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(128, 128,dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(128, 128,dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(128, 128,dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(128, 23,dtype=torch.float32)
        )

        self.valueHead = nn.Sequential(
            nn.Linear(122, 128,dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(128, 128,dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(128, 128,dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(128, 128,dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(128, 64,dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(64, 1,dtype=torch.float32),
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
            nn.Linear(122, 128,dtype=torch.float32),
            nn.LayerNorm(128,dtype=torch.float32),
            nn.ReLU(),
            # nn.Linear(128, 128,dtype=torch.float32),
            # nn.ReLU(),
            # nn.Linear(128, 128,dtype=torch.float32),
            # nn.ReLU(),
            # nn.Linear(128, 128,dtype=torch.float32),
            # nn.ReLU(),
            nn.Linear(128, 128,dtype=torch.float32),
            nn.LayerNorm(128,dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(128, 23,dtype=torch.float32)
        )

        self.valueHead = nn.Sequential(
            nn.Linear(122, 128,dtype=torch.float32),
            nn.LayerNorm(128,dtype=torch.float32),
            nn.ReLU(),
            # nn.Linear(128, 128,dtype=torch.float32),
            # nn.ReLU(),
            # nn.Linear(128, 128,dtype=torch.float32),
            # nn.ReLU(),
            # nn.Linear(128, 128,dtype=torch.float32),
            # nn.ReLU(),
            nn.Linear(128, 64,dtype=torch.float32),
            nn.LayerNorm(64,dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(64, 1,dtype=torch.float32),
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
        return len(self.children) > 0 or self.ischance

    def select(self):
        if self.ischance:
            dsp = expandable_moves
            outcome = r.choices(list(dsp.keys()),list(dsp.values()))[0]
            if not outcome in self.children:
                child_state = copy.deepcopy(self.state)
                child_state[0] = np.array(list(outcome),dtype=int)
                child = Node(self.game,self.args,child_state,self.active_player,self,False,iswhiteturn=True)
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
        # if self.ischance:
        #     for dices in expandable_moves:
        #         child_state = copy.deepcopy(self.state)
        #         child_state[0] = np.array([int(x) for x in dices])
        #         child = Node(self.game,self.args,child_state,(self.active_player+1)%len(self.state[-1]),self,iswhiteturn=True)
        #         index = ''.join(str(x) for x in dices)
        #         self.children[index] = child
        # else:
        last_turn = not self.iswhiteturn
        white_t = self.parent is None or self.parent.ischance
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

                child = Node(self.game, self.args, child_state,(self.active_player+1)%len(child_state[1]),self, act_mem, last_turn, white_t, prob)
                # if last_turn:
                #     child.expand(None)
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

class MCTSParallel:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model
    
    #returns the probabilities for the possible actions
    @torch.no_grad()
    def search(self,states,spGames,player,active_player_action,iswhiteturn):

        #dirichlet variant
        _, policy = self.model(torch.tensor(self.game.get_encoded_states(states),device=self.model.device,dtype=torch.float32))
        policy = torch.softmax(policy,1).detach().cpu().numpy()
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] * np.random.dirichlet([self.args['dirichlet_alpha']] * len(policy[0]), size=policy.shape[0])
        for i,spg in enumerate(spGames):
            spg.root = Node(self.game,self.args,states[i],player[i],None,active_player_action[i],False,iswhiteturn[i],visit_count=1)
            spg_policy = policy[i]

            node = spg.root
            valid_moves = self.game.get_valid_moves(node.state,node.active_player,node.iswhiteturn)
            for k in range(len(spg_policy)):
                if k not in valid_moves:
                    spg_policy[k] = 0

            spg_policy /= np.sum(spg_policy)

            spg.root.expand(spg_policy)


        # for i, spg in enumerate(spGames):
        #     spg.root = Node(self.game,self.args,states[i],player[i],None,active_player_action[i],False,iswhiteturn[i],visit_count=1)

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
                value, policy = self.model(torch.tensor(self.game.get_encoded_states(states),device=self.model.device,dtype=torch.float32))
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
            sample = memory[batchIdx:batchIdx+self.args['batch_size']] # Change to memory[batchIdx:batchIdx+self.args['batch_size']] in case of an error
            state, policy_targets, value_targets = zip(*sample)
            
            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(value_targets).reshape(-1, 1)
            
            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)
            
            out_value, out_policy = self.model(state)
            #print(out_policy.shape,"\n",policy_targets.shape,"\n",out_value.shape,"\n",value_targets.shape,"\n")
            assert torch.all(value_targets>=-1).item() and torch.all(value_targets<=1).item()
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
        for iteration in range(self.args['num_iterations']):
            memory = []
            print("Iteration ", iteration)
            self.model.eval()
            for selfPlay_iteration in tqdm(range(self.args['num_selfPlay_iterations']//self.args['num_parallel_games'])):
                memory += self.selfPlay()
                
            self.model.train()
            for epoch in tqdm(range(self.args['num_epochs'])):
                self.train(memory)

                if save_losses:
                    with open(f'diffRules/Models/Losses{loss_idx}/policy_loss.txt', 'a') as f:
                        f.write('%f \n' % np.average(policy_loss_arr))
                        f.close()
                    with open(f'diffRules/Models/Losses{loss_idx}/value_loss.txt', 'a') as f:
                        f.write('%f \n' % np.average(value_loss_arr))
                        f.close()
                    with open(f'diffRules/Models/Losses{loss_idx}/total_loss.txt', 'a') as f:
                        f.write('%f \n' % np.average(total_loss_arr))
                        f.close()
                policy_loss_arr.clear()
                value_loss_arr.clear()
                total_loss_arr.clear()
            
            torch.save(self.model.state_dict(), f"diffRules/Models/version_{loss_idx}_model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"diffRules/Models/version_{loss_idx}_optimizer_{iteration}.pt")
            # print("avg policy loss: ", np.average(policy_loss_arr))
            # print("avg value loss: ", np.average(value_loss_arr))
            # print("avg total loss: ", np.average(total_loss_arr))
            # policy_loss_arr.clear()
            # value_loss_arr.clear()
            # total_loss_arr.clear()

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
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    #model.load_state_dict(torch.load(f"Models/version_{loss_idx}_model_7.pt", map_location=device))
    #optimizer.load_state_dict(torch.load(f"Models/version_{loss_idx}_optimizer_7.pt", map_location=device))

    args = {
        'C': 2.5,
        'num_searches': 2500,
        'num_iterations': 8,
        'num_selfPlay_iterations': 1000,
        'num_parallel_games': 250,
        'num_epochs': 6,
        'batch_size': 64, 
        'temperature':1.3,
        'dirichlet_epsilon': 0.25,
        'dirichlet_alpha': 0.2
    }

    alphaZero = AlphaZeroParallel(model, optimizer, quixx, args)
    alphaZero.learn()

learning_rate = 0.01 #Losses{Anzahl der Nullen der lr} Bsp. lr = 0.001 -> Losses3
loss_idx = int(np.log10(learning_rate**-1))
policy_loss_arr, value_loss_arr, total_loss_arr = [], [], []
save_losses =  True
#delete current loss files
if save_losses:
    open(f'diffRules/Models/Losses{loss_idx}/policy_loss.txt', 'w').close()
    open(f'diffRules/Models/Losses{loss_idx}/value_loss.txt', 'w').close()
    open(f'diffRules/Models/Losses{loss_idx}/total_loss.txt', 'w').close()
testParallel()

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

    player_types = [P1model,P2model]
    quixx = simpleQ.Quixx()
    x = [[0,0]]
    ties = 0

    for i in range(num_games):
        state = quixx.get_initial_state()
        state[0] = simpleQ.dice()
        player = int(i > num_games/2)
        act_mem = None
        white_turn = True
        temp_player = player
        _, is_terminal = quixx.get_points_and_terminated(state)
        while not is_terminal:
            v = quixx.get_valid_moves(state,temp_player,white_turn)
            if not player_types[temp_player] is None:
                _, policy = player_types[temp_player](torch.tensor(quixx.get_encoded_state(state),device=player_types[temp_player].device,dtype=torch.float32))
                policy = torch.softmax(policy,0).detach().cpu().numpy()

                for i in range(len(policy)):
                    if i not in v:
                        policy[i] = 0
                    
                assert np.sum(policy) > 0

                policy /= np.sum(policy)
                action = np.argmax(policy)
            else:
                action = r.choice(v)
            
            if act_mem is None:
                act_mem = action
            elif temp_player == player:
                action = -1 if action+act_mem == 0 else action

            state = quixx.get_next_state(state,temp_player,action)
            points, is_terminal = quixx.get_points_and_terminated(state)
            if not white_turn:
                player = (player+1)%2
                temp_player = player
                act_mem = None
                white_turn = True
                state[0] = simpleQ.dice()
            else:
                temp_player = (temp_player+1)%2
                if temp_player == player:
                    white_turn = False

        if points[0] != points[1]:
            x = np.append(x,[points],axis=0)
        else:
            ties += 1
    return 1-(np.sum(np.argmax(x[1:],axis=1))/(num_games-ties)),ties


def img_for_simulation():
    for version in range(2,5):
        winrates = []
        for i in tqdm(range(8)):
            winrates = np.append(winrates,simulate(1000,i,None,version)[0]*100)
        plt.plot(np.arange(8), winrates, label = f"lr = {10**(-version)}")

    plt.title('Quixx winrate against random Bot')
    plt.xlabel('Iteration')
    plt.ylabel('Win rate in %')
    plt.legend(loc="lower right")
    plt.show()

#img_for_simulation()
