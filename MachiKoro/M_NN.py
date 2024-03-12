import numpy as np
from numpy import random
import itertools as iter
import random as r
import copy
from tqdm import tqdm
import math
import torch
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt

import machikoro

print(torch.__version__)

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
            nn.Linear(155, 128,dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(128, 64,dtype=torch.float32),#128 128
            nn.ReLU(),
            nn.Linear(64, 24,dtype=torch.float32)
        )

        self.valueHead = nn.Sequential(
            nn.Linear(155, 128,dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(128, 64,dtype=torch.float32),#64 64
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
            nn.Linear(155, 128,dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(128, 128,dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(128, 128,dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(128, 128,dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(128, 64,dtype=torch.float32),#128 128
            nn.ReLU(),
            nn.Linear(64, 24,dtype=torch.float32)
        )

        self.valueHead = nn.Sequential(
            nn.Linear(155, 128,dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(128, 128,dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(128, 128,dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(128, 128,dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(128, 64,dtype=torch.float32),#64 64
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
            nn.Linear(155, 128,dtype=torch.float32),
            nn.LayerNorm(128,dtype=torch.float32),
            nn.ReLU(),
            # nn.Linear(128, 128,dtype=torch.float32),
            # nn.ReLU(),
            # nn.Linear(128, 128,dtype=torch.float32),
            # nn.ReLU(),
            # nn.Linear(128, 128,dtype=torch.float32),
            # nn.ReLU(),
            nn.Linear(128, 64,dtype=torch.float32),#128 128
            nn.LayerNorm(128,dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(64, 24,dtype=torch.float32)
        )

        self.valueHead = nn.Sequential(
            nn.Linear(155, 128,dtype=torch.float32),
            nn.LayerNorm(128,dtype=torch.float32),
            nn.ReLU(),
            # nn.Linear(128, 128,dtype=torch.float32),
            # nn.ReLU(),
            # nn.Linear(128, 128,dtype=torch.float32),
            # nn.ReLU(),
            # nn.Linear(128, 128,dtype=torch.float32),
            # nn.ReLU(),
            nn.Linear(128, 64,dtype=torch.float32),#64 64
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

class Node:

    def __init__(self, game, args, state, active_player, parent=None, action_taken=None, ischance = False, dices = None, has_used_rethrow = False, prior=0, visit_count=0):
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
        self.visit_count = visit_count
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
                d = [int(x) for x in dices]
                child_state = copy.deepcopy(self.state)
                can_rethrow = self.state[2][self.active_player][3]
                if not can_rethrow or self.has_used_rethrow:
                    self.game.distribution(child_state,self.active_player,np.array(d))
                child = Node(self.game,self.args,child_state,self.active_player,self,None,False,np.array(d),self.has_used_rethrow)
                self.children[dices] = child
        else:
            for action, prob in enumerate(policy):
                if prob > 0:
                    child_state = copy.deepcopy(self.state)
                    if action <= 19:
                        child_state = self.game.get_next_state(child_state,self.active_player,action)
                        assert self.dices is not None
                        #case: Player doesn't have upgrade 1 (dice choice), so the child doesn't need to decide on num of dice because only one option possible
                        #the following block can be skipped since upgrade 3 (extra turn) is only possible with 2 dice thus needing upgrade 1 (dice choice)
                        if child_state[2][self.active_player][0] == 0:
                            child = Node(self.game, self.args, child_state, (self.active_player+1) % len(child_state[0]), self, action, True, np.array([0]), False, prob)
                            child.expand(None)
                        else:
                            if  child_state[2][self.active_player][2] and (np.count_nonzero(self.dices == np.max(self.dices)) == 2):
                                p = self.active_player
                            else:
                                p = (self.active_player+1) % len(child_state[0])

                            child = Node(self.game,self.args,child_state,p, self, action, False, None, False, prob)
                    elif action == 22:
                        self.game.distribution(child_state,self.active_player,np.array([int(x) for x in self.dices]))
                        child = Node(self.game,self.args,child_state,self.active_player,self,action,False,self.dices,False,prob)
                    else:#action in (20,21,23)
                        dice = np.array([0]) if action == 20 else np.array([0,0]) if action == 21 else self.dices
                        child = Node(self.game, self.args, child_state, self.active_player, self, action, True, dice, action==23, prob)
                        child.expand(None)
                    self.children[action] = child
      
    # def backpropagate(self,value):
    #     self.value_sum += value 
    #     self.visit_count += 1
        
    #     if self.parent is not None:
    #         if self.active_player != self.parent.active_player:
    #             value = -value
    #         self.parent.backpropagate(value) 
    def backpropagate(self,value):#iterativ backpropagation since recursive threw error
        node = self
        while not node is None:
            node.value_sum += value 
            node.visit_count += 1
            
            if node.parent is not None and node.active_player != node.parent.active_player:
                    value = -value
            node = node.parent

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

#parallel games
class MCTSParallel:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model
    
    #returns the probabilities for the possible actions
    @torch.no_grad()
    def search(self,states,spGames,player,last_action,dices=None):
        
        #dirichlet variant
        _, policy = self.model(torch.tensor(self.game.get_encoded_states(states),device=self.model.device))
        policy = torch.softmax(policy,1).detach().cpu().numpy()
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] * np.random.dirichlet([self.args['dirichlet_alpha']] * len(policy[0]), size=policy.shape[0])
        for i,spg in enumerate(spGames):
            spg.root = Node(self.game,self.args,states[i],player[i],None,last_action[i],False,dices[i],visit_count=1)
            spg_policy = policy[i]

            node = spg.root
            if node.action_taken is not None and node.action_taken <= 19:
                    spg_policy[:20] = 0
                    spg_policy[22:] = 0
            elif node.action_taken is None and node.state[2][node.active_player][3] and (node.parent is None or node.has_used_rethrow == False):
                spg_policy[:22] = 0                  
            else:
                valid_moves = self.game.get_valid_moves(node.state,node.active_player)
                for i in range(len(spg_policy)):
                    if i not in valid_moves:
                        spg_policy[i] = 0

            spg_policy /= np.sum(spg_policy)

            spg.root.expand(spg_policy)

        #original
        # for i,spg in enumerate(spGames):
        #     spg.root = Node(self.game,self.args,states[i],player[i],None,last_action[i],False,dices[i],visit_count=1)

        for search in range(self.args['num_searches']):
            for i, spg in enumerate(spGames):
                spg.node = None
                node = spg.root

                #no search needed if only one possible action 
                if len(spg.root.children) == 1:
                    for child_key, child_value in spg.root.children.items():
                        child_value.visit_count += 1
                    continue

                while node.is_fully_expanded():
                    node = node.select()
                
                winner, is_terminal = self.game.is_terminated(node.state)
                value = -1**(winner)

                if is_terminal:
                    node.backpropagate(value)
                else:
                    spg.node = node

            expandable_spGames = [mappingIdx for mappingIdx in range(len(spGames)) if spGames[mappingIdx].node is not None]
            if len(expandable_spGames) > 0:
                states = np.stack([spGames[mappingIdx].node.state for mappingIdx in expandable_spGames])
                value, policy = self.model(torch.tensor(self.game.get_encoded_states(states),device=self.model.device,dtype=torch.float32))
                policy = torch.softmax(policy,1).detach().cpu().numpy()
                value = value.cpu().detach().numpy()

            for i, mappingIdx in enumerate(expandable_spGames):
                node = spGames[mappingIdx].node
                spg_policy, spg_value = policy[i], value[i]
                #for this case we don't have to check for the upgrade because of the expand method:
                #if a new turn starts without the first upgrade the expand method append a chance node thus select in this method leads to a node with action_taken=None
                #else if a player has the upgrade the expand method appends a normal child with action <= 19
                if node.action_taken is not None and node.action_taken <= 19:
                    spg_policy[:20] = 0
                    spg_policy[22:] = 0
                #parent of a possible rethrow decision node is always a chance node:
                    #children of chance nodes have always action_taken = None and if before a chance node a rethrow decision was used (action 23) then the chance node
                    #has the attribute has_used_rethrow=True
                elif node.action_taken is None and node.state[2][node.active_player][3] and (node.parent is None or node.has_used_rethrow == False):
                    spg_policy[:22] = 0                  
                else:
                    valid_moves = self.game.get_valid_moves(node.state,node.active_player)
                    for i in range(len(spg_policy)):
                        if i not in valid_moves:
                            spg_policy[i] = 0

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
        action = len(spGames)*[None]
        dices = len(spGames)*[None]
        dice_choice_available = len(spGames)*[False]
        rethrow_choice_available = len(spGames)*[False]
        zero_cnt = len(spGames)*[0]
        while len(spGames) > 0:
            states = np.stack([spg.state for spg in spGames])
            for i in range(len(spGames))[::-1]:
                if not dice_choice_available[i] and not rethrow_choice_available[i]:
                    if dices[i] is None:
                        dices[i] = machikoro.dice(1)
                    self.game.distribution(states[i],player[i],dices[i])
                    assert np.all(states[i][0]<64)

                elif dices[i] is None and not dice_choice_available[i] and rethrow_choice_available[i]:
                    dices[i] = machikoro.dice(1)
                
                #special case:
                # while none of these upgrades are available the search method would be called with action <= 19 which leads to a search with action_probs
                # at 100% for dice choice one (20) since the other option is not unlocked yet 
                if action[i] is not None and action[i] <= 19 and not dice_choice_available[i] and not rethrow_choice_available[i]:
                    action[i] = 99

            self.mcts.search(states,spGames,player,action,dices)

            for i in range(len(spGames))[::-1]:
                spg = spGames[i]

                action_probs = np.zeros(24)
                for child_key, child_value in spg.root.children.items():
                    action_probs[child_key] = child_value.visit_count

                assert np.sum(action_probs)>0

                action_probs /= np.sum(action_probs)
                #print(action_probs)
                spg.memory.append((spg.root.state, action_probs, player[i]))

                temperature_action_probs = action_probs ** (1 / self.args['temperature'])#squishes the values together to allow more exploration
                if dice_choice_available[i]:

                    action[i] = r.choices(np.arange(len(action_probs)),temperature_action_probs)[0]
                    dices[i] = machikoro.dice(action[i]-19)                   

                    dice_choice_available[i] = False                  
                elif rethrow_choice_available[i]:

                    action[i] = r.choices(np.arange(len(action_probs)),temperature_action_probs)[0] 

                    dices[i] = machikoro.dice(len(dices[i])) if (action[i] - 22) else dices[i]

                    rethrow_choice_available[i] = False
                else:

                    action[i] = r.choices(np.arange(len(action_probs)),temperature_action_probs)[0]

                    spg.state = self.game.get_next_state(spg.state,player[i],action[i])

                    winner, is_terminal = self.game.is_terminated(spg.state)

                    if is_terminal or zero_cnt[i] == 14:
                        for hist_neutral_state, hist_action_probs, hist_player in spg.memory:
                            hist_outcome = 1 if hist_player == winner else -1 #if zero_cnt[i] > 14 else 
                            return_memory.append((
                                self.game.get_encoded_state(hist_neutral_state),
                                hist_action_probs,
                                hist_outcome
                            ))

                        del spGames[i]
                        del player[i]
                        del action[i]
                        del dices[i]
                        del dice_choice_available[i]
                        del rethrow_choice_available[i]
                        del zero_cnt[i]
                        continue
                                                             
                    #zero_cnt[i] =  zero_cnt[i]+1 if action[i] == 0 else 0 if action[i] <= 19 else zero_cnt[i]
                    
                    if action[i] in range(0,20):
                        # if doublets thrown and upgrade 3 available the player gets another chance
                        if spg.state[2][player[i]][2] and (np.count_nonzero(dices[i] == np.max(dices[i])) == 2):
                            player[i] = player[i]
                        else:
                            player[i] = (player[i] +1) % len(spg.state[0])
                        dice_choice_available[i] = spg.state[2][player[i]][0]
                        rethrow_choice_available[i] = spg.state[2][player[i]][3]
                        dices[i] = None

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
            #print("targets: ", policy_targets)
            out_value, out_policy = self.model(state)
            assert torch.all(torch.any(policy_targets>0,-1))
            assert torch.all(value_targets>=-1).item() and torch.all(value_targets<=1).item() ###delete tanh
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
                
            # print("Memory")
            # for a,b,c in memory:
            #     print("state: ", a)
            #     print("policy target: ", np.argsort(b), "cnt: ",np.count_nonzero(b))
            #     print("value target: ", c)

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

            torch.save(self.model.state_dict(), f"ModelsNN3/version_{loss_idx}_model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"ModelsNN3/version_{loss_idx}_optimizer_{iteration}.pt")

            #get average loss
            # print("avg policy loss: ", np.average(policy_loss_arr))
            # print("avg value loss: ", np.average(value_loss_arr))
            # print("avg total loss: ", np.average(total_loss_arr))
            # policy_loss_arr.clear()
            # value_loss_arr.clear()
            # total_loss_arr.clear()


class SPG:
    def __init__(self,game):
        self.state = game.get_initial_state(2)
        self.memory = []
        self.root = None
        self.node = None

def testParallel():
    mk = machikoro.Machikoro()
    model = NeuralNetwork(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)#0.001

    # model.load_state_dict(torch.load('Models/model_2.pt', map_location=device))
    # optimizer.load_state_dict(torch.load('Models/optimizer_2.pt', map_location=device))
    args = {
        'C': 2.5,
        'num_searches': 500,#500            800
        'num_iterations': 8,
        'num_selfPlay_iterations': 300,#450  
        'num_parallel_games': 100,#150      
        'num_epochs': 6,
        'batch_size': 64,#64
        'temperature': 1.3,
        'dirichlet_epsilon': 0.25,
        'dirichlet_alpha': 0.15
    }

    alphaZero = AlphaZeroParallel(model, optimizer, mk, args)
    alphaZero.learn()

learning_rate = 0.0001
loss_idx = int(np.log10(learning_rate**-1))
policy_loss_arr, value_loss_arr, total_loss_arr = [], [], []
save_losses = True
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
    winr = []
    mk = machikoro.Machikoro()
    ties = 0
    for i in range(num_games):
        player = int(i > num_games/2)
        state = mk.get_initial_state(2)

        action = None
        dices = machikoro.dice(1)
        dice_choice_available = False
        rethrow_choice_available = False
        zero_cnt = 0

        while True:
            
            if not player_types[player] is None:
                _,policy = player_types[player](torch.tensor(mk.get_encoded_state(state),device=player_types[player].device,dtype=torch.float32))
                policy = torch.softmax(policy,0).detach().cpu().numpy()

            if dice_choice_available:
                if not player_types[player] is None:
                    policy[:20] = 0
                    policy[22:] = 0
                    policy /= np.sum(policy)
                    action = np.argmax(policy)
                else:
                    action = r.choice([20,21])
                dices = machikoro.dice(action-19)
                dice_choice_available = False
            elif rethrow_choice_available:
                if not player_types[player] is None:
                    policy[:22] = 0
                    policy /= np.sum(policy)
                    action = np.argmax(policy)
                else:
                    action = r.choice([22,23])
                
                dices = machikoro.dice(len(dices)) if action-22 else dices
                rethrow_choice_available = False
            else:
                mk.distribution(state,player,dices)
                v = mk.get_valid_moves(state,player)
                if not player_types[player] is None:
                    for i in range(len(policy)):
                        if i not in v:
                            policy[i] = 0
                    policy /= np.sum(policy)
                    action = np.argmax(policy)
                else:
                    action = r.choice(v)

                state = mk.get_next_state(state,player,action)

                winner, is_terminal = mk.is_terminated(state)

                if is_terminal or zero_cnt == 25:
                    if zero_cnt <25:
                        winr = np.append(winr,winner)
                    else:
                        ties +=1
                    break

                if action == 0:
                    zero_cnt += 1
                else:
                    zero_cnt = 0
                
                if action in range(20):
                    if len(dices)==2 and state[2][player][2] and dices[0]==dices[1]:
                        player = player
                    else:
                        player = (player +1) % len(state[0])
                    dice_choice_available = state[2][player][0]
                    rethrow_choice_available = state[2][player][3]
                    dices = machikoro.dice(1)
    return 1-(np.sum(winr)/(num_games-ties)),ties

def img_for_simulation():
    for version in range(2,5):
        winrates = []
        t = []
        for i in tqdm(range(8)):
            winr, ties = simulate(1000,i,None,version)
            winrates = np.append(winrates,winr*100)
            t = np.append(t,ties)
        plt.plot(np.arange(8), winrates, label = f"lr = {10**(-version)}")
        print(ties)

    plt.title('Machikoro winrate against random Bot')
    plt.xlabel('Iteration')
    plt.ylabel('Win rate in %')
    plt.legend(loc="lower right")
    plt.show()

#img_for_simulation()