import numpy as np
from numpy import random
import random as r
import time
import math
import itertools as iter
from tqdm import tqdm

class quixx:

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
        for j in range(1,len(state)-1):
            if np.sum([state[j][i][-1]==1 for i in range(4)])>=2 or np.any([state[-1][k]==4 for k in range(num_players)]):
                terminated = True
        
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

Quixx = quixx()

x = np.array([[0,0,0,0]])
st = time.process_time()
for i in tqdm(range(10000)):
    game = Quixx.get_initial_state(4)
    num_players = len(game)-2
    points, terminated = Quixx.get_points_and_terminated(game)
    player = 0
    while not terminated:
        game[0] = dice()
        #white_dice
        act = greedy_bot(Quixx,game,Quixx.get_valid_moves(game,player,True),player)
        game = Quixx.get_next_state(game,player,act)
        play = (player+1)%num_players
        while play != player:
            act_r = greedy_bot(Quixx,game,Quixx.get_valid_moves(game,play,True),player)
            game = Quixx.get_next_state(game,play,act_r)
            play = (play+1)%num_players
        #rest of turn
        act_2 = greedy_bot(Quixx,game,Quixx.get_valid_moves(game,player),player)
        #if the sum is >0 at least one combination was used
        if act + act_2 == 0:
            act_2 = -1
        game = Quixx.get_next_state(game,player,act_2)
        player = (player+1)%num_players
        points, terminated = Quixx.get_points_and_terminated(game)

    x = np.append(x,[points],axis=0)
et = time.process_time()
res = et - st
# print(dict)
print('CPU Execution time:', res, 'seconds')
x = x[1:]
print(np.max(x,axis=0))
print(np.min(x,axis=0))
print(np.average(x,axis=0))
print(np.median(x,axis=0))