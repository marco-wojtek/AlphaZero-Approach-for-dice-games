import numpy as np
from numpy import random
import random as r
import time
import math
import itertools as iter
from tqdm import tqdm
import copy

class Quixx:

    def __init__(self):
        #contains the indices for all spaces
        self.action_space = np.reshape(np.arange(1,45),(4,11))
        self.point_space = np.array([0,1,3,6,10,15,21,28,36,45,55,66,78])

    def get_initial_state(self,player_num):
        assert player_num in [2,3,4]
        arr = np.array([
            np.arange(2,13),
            np.arange(2,13),
            np.arange(12,1,-1),
            np.arange(12,1,-1)
            ])
        if player_num == 2:
            return np.array([
                np.zeros(6), 
                arr.copy(),
                arr.copy(),
                np.zeros(2)],dtype=object)
        elif player_num == 3:
            return np.array([
                np.zeros(6), 
                arr.copy(),
                arr.copy(),
                arr.copy(),
                np.zeros(3)],dtype=object)
        else:
            return np.array([
                np.zeros(6), 
                arr.copy(),
                arr.copy(),
                arr.copy(),
                arr.copy(),
                np.zeros(4)],dtype=object)

    #action 0 means "no mark"
    #marks the number of the action and sets all values between the last mark and chosen mark to 0 
    #since one dice option can be freely skipped the action 0 has no influence but the special input -1 enters the error 
    def get_next_state(self,state,player,action):
        if action == -1:
            state[-1][player] +=1
        elif action >0:    
            row = np.argwhere(self.action_space==action)[0][0]
            val = np.argwhere(self.action_space==action)[0][1]
            marked_num = np.argwhere(state[player+1][row]==1)
            last_marked = marked_num[-1][0] if len(marked_num) >0 else -1
            state[player+1][row][val] = 1
            state[player+1][row][last_marked+1:val] = 0 
        return state

    #resturns a number of possible moves; for white dice roatation max. 5 (one is no choice); coloured dice has max. 9 (one is no choice) error count is handled out
    #the values returned are the values of the action_space with the additional option 0
    def get_valid_moves(self,state,player,white_rotation=False):
        options = np.array([0])
        if white_rotation:
            dice_value = state[0][0]+state[0][1]
            for k in range(4):
                x = np.argwhere(state[player+1][k]== dice_value)
                if len(x)>0:
                    options = np.append(options,self.action_space[k][x])
        else:
            for k in range(4):
                dice_val_1 = state[0][0] + state[0][2+k]
                dice_val_2 = state[0][1] + state[0][2+k]
                x_1 = np.argwhere(state[player+1][k]== dice_val_1)
                x_2 = np.argwhere(state[player+1][k]== dice_val_2)
                if len(x_1)>0:
                    options = np.append(options,self.action_space[k][x_1])
                if len(x_2)>0:
                    options = np.append(options,self.action_space[k][x_2])
        #filter out the last element of the row if not 5 were marked before
        closed = np.array([11,22,33,44]) # The closing row actions
        for n in range(len(closed)):
            cnt = np.count_nonzero(state[player+1][n][np.where(state[player+1][n]==1)]) #count available number of ones
            if cnt < 5:
                index = np.argwhere(options==closed[n])
                options = np.delete(options,index)
        return np.unique(options)

    #check for >=2 closed rows for each player, a row is closed if the last number in a row is marked 
    def get_points_and_terminated(self,state):
        num_players = len(state)-2
        terminated = False
        closed_rows = [False,False,False,False]
        for j in range(1,len(state)-1):
            arr = [state[j][i][-1]==1 for i in range(4)]
            closed_rows = closed_rows or arr
            if np.sum(arr)>=2 or np.any([state[-1][k]==4 for k in range(num_players)]):
                terminated = True
        
        #for every closed row mark all other players numbers of that row with 0
        for j in range(1,len(state)-1):
            for i in range(4):
                if closed_rows[i]:
                    marked_num = np.argwhere(state[j][i]==1)
                    last_marked = marked_num[-1][0] if len(marked_num) >0 else 0
                    state[j][i][last_marked:] = 0

        points = np.zeros(num_players)
        for n in range(num_players):
            for m in range(4):
                x = state[n+1][m][-1] == 1
                points[n] += self.point_space[np.count_nonzero(state[n+1][m][np.where(state[n+1][m]==1)])+x]
            points[n] -= 5* state[-1][n]

        return points, terminated

def dice():
    return random.randint(1,7,size=(6))

def random_bot(valid_moves):
    return r.choice(valid_moves)

#greedy bot tries to make as many marks as possible per turn thus trying to avoid the 0 option which is always on index 0
def greedy_bot(game,state,valid_moves,player):
    last_marked = np.array([])
    for i in range(4):
        x = np.argwhere(state[player+1][i]==1)
        x = game.action_space[i][x[-1]] if len(x)>0 else game.action_space[i][0]
        last_marked = np.append(last_marked,x)
    distances = np.array([])
    for k in range(1,len(valid_moves)):
        index = int(valid_moves[k]/11) if valid_moves[k] not in [11,22,33,44] else (int(valid_moves[k]/11) -1)     
        distances = np.append(distances,valid_moves[k]-last_marked[index])
    distances = np.append(np.inf,distances)
    return valid_moves[np.argmin(distances)]

# player = 0
# state[0] = dice()
# v = Quixx.get_valid_moves(state,player)
# w = Quixx.get_valid_moves(state,player,True)
# print(v)
# print(w)
# state = Quixx.get_next_state(state,player,r.choice(v))
# print(state)
# print(Quixx.action_space)
# x = np.array([[0,0,0,0]])
# st = time.process_time()
# for i in tqdm(range(10000)):
#     game = Quixx.get_initial_state(4)
#     num_players = len(game)-2
#     points, terminated = Quixx.get_points_and_terminated(game)
#     player = 0
#     while not terminated:
#         game[0] = dice()
#         #white_dice
#         act = greedy_bot(Quixx,game,Quixx.get_valid_moves(game,player,True),player)
#         game = Quixx.get_next_state(game,player,act)
#         play = (player+1)%num_players
#         while play != player:
#             act_r = greedy_bot(Quixx,game,Quixx.get_valid_moves(game,play,True),player)
#             game = Quixx.get_next_state(game,play,act_r)
#             play = (play+1)%num_players
#         #rest of turn
#         act_2 = greedy_bot(Quixx,game,Quixx.get_valid_moves(game,player),player)
#         #if the sum is >0 at least one combination was used
#         if act + act_2 == 0:
#             act_2 = -1
#         game = Quixx.get_next_state(game,player,act_2)
#         player = (player+1)%num_players
#         points, terminated = Quixx.get_points_and_terminated(game)

#     x = np.append(x,[points],axis=0)
# et = time.process_time()
# res = et - st
# # print(dict)
# print('CPU Execution time:', res, 'seconds')
# x = x[1:]
# print(np.max(x,axis=0))
# print(np.min(x,axis=0))
# print(np.average(x,axis=0))
# print(np.median(x,axis=0))

class Node:

    def __init__(self, game, args, state, active_player, parent=None, action_taken=None, active_player_action = None, ischance = False, iswhiteturn = False):
        self.game = game
        self.state = state
        self.args = args
        self.active_player = active_player
        self.parent = parent
        self.children = {}

        self.action_taken = action_taken
        self.active_player_action = active_player_action
        self.ischance = ischance

        self.iswhiteturn = iswhiteturn
        self.expandable_moves = game.get_valid_moves(self.state,self.active_player,self.iswhiteturn)

        self.visit_count = 0
        self.value_sum = 0

    def is_fully_expanded(self):
        if self.ischance:
            return len(self.expandable_moves[0]) == len(self.children) and len(self.children) > 0
        
        return len(self.expandable_moves) == len(self.children) and len(self.children) > 0
    
    def select(self):
        if self.ischance:
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
    
    def get_ucb(self,child):
        #q_value is normalised from [-1,1] 
        if self.active_player==child.active_player:
            q_value = ((child.value_sum / child.visit_count) + 1) / 2
        else:#if the child has a different active player the q value is inverted because the score is from a different view
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.args['C'] * math.sqrt(math.log(self.visit_count) / child.visit_count)
    
    def expand(self):
        #expansion for chance nodes
        #expansion for white dice and coloured dice
        if self.ischance:
            for dices in self.expandable_moves[0]:
                child_state = copy.deepcopy(self.state)
                child_state[0] = np.asarray(dices)
                child = Node(self.game,self.args,self.state,(self.active_player+1)%len(self.state[2]),self,iswhiteturn=True)
                self.children[dices] = child
        elif self.iswhiteturn:
            action = self.expandable_moves[r.choice(np.where(self.expandable_moves!=-1)[0])]
            child_state = copy.deepcopy(self.state)
            if self.parent is None or self.parent.ischance:#players own white turn
                child = Node(self.game,self.args,self.state,(self.active_player+1)%len(self.state[2]),self,action,action,False,True)
            else:#own action doesn't matter so the parents action is parsed through
                child = Node(self.game,self.args,self.state,(self.active_player+1)%len(self.state[2]),self,action,self.action,False,False)
            self.children[action] = child
            self.expandable_moves[np.argwhere(self.expandable_moves==action)[0][0]]  = -1
        else:
            action = self.expandable_moves[r.choice(np.where(self.expandable_moves!=-1)[0])]
            child_state = copy.deepcopy(self.state)
            #add error if no action other than 0 is taken
            child_state[2][self.active_player] += ((self.active_player_action + action)==0)
            child = Node(self.game,self.args,self.state,(self.active_player+1)%len(self.state[2]),self,action,None,True,False)
            child.expand()#add chance outcomes
            self.children[action] = child
            self.expandable_moves[np.argwhere(self.expandable_moves==action)[0][0]]  = -1

        return child
    
    def simulate(self): ## RETHINK FOR EFFICENCY
        points, is_terminal = self.game.get_points_and_terminated(self.state)
        value = np.argmax(points) if np.count_nonzero(points==np.max(points))==1 else -1
        if is_terminal:
            return value
        
        rollout_state = copy.deepcopy(state)
        player = self.active_player

        #do one rotation with specific chosen action
        rollout_state = self.game.get_next_state(rollout_state,player,self.action_taken)
        player = (player+1)%len(rollout_state[2])
        points, is_terminal = self.game.get_points_and_terminated(self.state)
        value = np.argmax(points) if np.count_nonzero(points==np.max(points))==1 else -1
        if is_terminal:
            return value
        if self.parent != None and self.parent.ischance and self.iswhiteturn:       
            v = self.game.get_valid_moves(rollout_state,player,True)
            action = r.choice(v)
            rollout_state = self.game.get_next_state(rollout_state,player,action)
            player = (player+1)%len(rollout_state[2])
            points, is_terminal = self.game.get_points_and_terminated(self.state)
            value = np.argmax(points) if np.count_nonzero(points==np.max(points))==1 else -1
            if is_terminal:
                return value
            v = self.game.get_valid_moves(rollout_state,player)
            action = r.choice(v)
            rollout_state = self.game.get_next_state(rollout_state,player,action)
            player = (player+1)%len(rollout_state[2])
        elif self.parent != None and self.parent.iswhiteturn and self.iswhiteturn:
            v = self.game.get_valid_moves(rollout_state,player)
            action = r.choice(v)
            rollout_state = self.game.get_next_state(rollout_state,player,action)
            player = (player+1)%len(rollout_state[2])

        points, is_terminal = self.game.get_points_and_terminated(self.state)
        value = np.argmax(points) if np.count_nonzero(points==np.max(points))==1 else -1
        if is_terminal:
            return value
        
        white_cnt = 0
        while not is_terminal:
            v = self.game.get_valid_moves(rollout_state,player,white_cnt<2)
            action = r.choice(v)
            rollout_state = self.game.get_next_state(rollout_state,player,action)
            player = (player+1)%len(rollout_state[2])
            if white_cnt == 2:
                white_cnt = 0
            points, is_terminal = self.game.get_points_and_terminated(self.state)
        value = np.argmax(points) if np.count_nonzero(points==np.max(points))==1 else -1
        return value
    
    def backpropagate(self,value):
        self.value_sum += (-1)**(value!=self.active_player) * (value>=0) 
         
        if self.parent is not None:
            self.parent.backpropagate(value) 

quixx = Quixx()
state = quixx.get_initial_state(2)
state[0] = dice()
print(state)
print(quixx.get_valid_moves(state,0,False))