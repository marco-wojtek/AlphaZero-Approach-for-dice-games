import numpy as np
from numpy import random
import random as r
import time
import itertools as iter
import copy

#returns a numpy array with num_dice values between 1 and 6 both included
def dice_throw(num_dice=5):
    assert(type(num_dice) == int)
    return random.randint(1,7,size=(num_dice))

#returns the dice with changed values; to_rethow must be a list with True or False values
def rethrow(dice, to_rethrow):
    assert(len(dice) == len(to_rethrow))
    for i in range(len(dice)):
        if to_rethrow[i]:
            dice[i] = random.randint(1,6) 
    return dice

#all functions to calculate how many points the dice returns in the specific category

#return True if dice contains a full House
def full_house(dice):
    i = [False,False]
    for n in range(1,7):
        if np.count_nonzero(dice==n) == 3:
            i[0] = True
        elif np.count_nonzero(dice==n) == 2:
            i[1] = True
    return i[0] and i[1]

#returns two booleans wether dice contains a small or large straight
def straight(dice):
    sorted_dice = np.unique(dice)
    length = len(sorted_dice)
    #if sorted dices are only 3 or less distinct values no straight is possible
    if length < 4:
        return [False,False]
    cnt_longest_seq = 1
    for n in range(1,length):
        if sorted_dice[n] == sorted_dice[n-1]+1:
            cnt_longest_seq += 1
    return [True,False] if cnt_longest_seq == 4 else [True,True] #length can only be 4 or 5

#all points can be set/adjusted here
count_eyes = lambda x, i: np.count_nonzero(x==i)*i
three_same = lambda x: np.sum(x) if any([np.count_nonzero(x==i)==3  for i in range(1,6)]) else 0
four_same = lambda x: np.sum(x) if any([np.count_nonzero(x==i)==4  for i in range(1,6)]) else 0
fullHouse = lambda x: 25 if full_house(x) else 0
small_straight = lambda x: 30 if straight(x)[0] else 0    
large_straight = lambda x: 40 if straight(x)[1] else 0
yahtzee = lambda x: 50 if np.count_nonzero(x==x[0]) == 5 else 0
chance = lambda x: np.sum(x)

option_names = np.array([
                "Rethrow",
                "Ones",
                "Twos",
                "Threes",
                "Fours",
                "Fives",
                "Sixes",
                "Three Same (All Eyes)",
                "Four Same (All Eyes)",
                "Full House(25p)",
                "small Straight (25p)",
                "large Straight (30p)",
                "Yahtzee (50p)",
                "Chance (all Eyes)"])

options = np.array([lambda x: count_eyes(x,i=1),#0
           lambda x: count_eyes(x,i=2),#1
           lambda x: count_eyes(x,i=3),#2
           lambda x: count_eyes(x,i=4),#3
           lambda x: count_eyes(x,i=5),#4
           lambda x: count_eyes(x,i=6),#5
           three_same,#6
           four_same,#7
           fullHouse,#8
           small_straight,#9
           large_straight,#10
           yahtzee,#11
           chance#12
           ])

class yahtzee:
    def __init__(self,num_players):
        assert num_players in [1,2,3,4]
        self.player_num = num_players

    def get_valid_moves(self,state,player,rethrow):
        return np.append(rethrow<2,state[1][player] == -1)

    def get_next_state(self,state,player,action,re_dice=[0,0,0,0,0]):
        if action == 0:
            state[0] = rethrow(state[0],re_dice)
        else:
            state[1][player][action-1] = options[action-1](state[0])
        return state

    def get_initial_state(self):
        state = np.array([
            np.zeros(5),
            np.ones((self.player_num,13)) * -1
        ],dtype=object)
        return state

    def get_points_and_terminated(self,state):
        points = np.zeros(len(state[1]))
        for i in range(len(state[1])):
            x = 35 if np.sum(state[1][i][:6][state[1][i][:6]!=-1]) >= 63 else 0 #Bonus if the sum of the first 6 is greater than 63
            points[i] = np.sum(state[1][i][state[1][i]!=-1])
        if np.any(state[1]==-1):
            return points,False
        return points, True

# Yahtzee = yahtzee(2)
# game = Yahtzee.get_initial_state()
# game = Yahtzee.get_next_state(game,0,0,[1,1,1,1,1])
# print(len(game[0]))
# player = 0
# throw = 0
# while not get_points_and_terminated(0,game)[1]:
#     print("Player {} turn".format(player))
#     print(game[0])
#     print(np.c_[option_names[get_valid_moves(0,game,player,throw)],np.argwhere(get_valid_moves(0,game,player,throw))])
#     action = int(input("Player's choice: "))
#     if action == 0:
#         dice = input("Rethrow dice as a b c d e with 1 for rethrow and 0 for not: ")
#         dice = np.array([int(temp) for temp in dice.split() if temp.isdigit()])
#         game = get_next_state(0,game,player,action,dice)
#         throw += 1
#     else:
#         dice = np.array([1,1,1,1,1])
#         game = get_next_state(0,game,player,action,dice)
#         throw = 0
#         player = (player + 1) % len(game[1])
# print(game[1])
# points, t = get_points_and_terminated(0,game)
# print("Winner is Player {} with {} Points".format(np.argmax(points),np.max(points)))

all_permutations = list(iter.product(range(0,2),repeat=5))

def random_bot_action(game,state,player,valid_actions):
    c = r.choice(np.argwhere(valid_actions))[0]
    throw = 1
    while c==0:
        state = game.get_next_state(state,player,0,np.asarray(r.choice(all_permutations)))
        v = game.get_valid_moves(state,player,throw)
        c = r.choice(np.argwhere(v))[0]
        throw += 1
    return c

def greedy_bot_action(game,state,player,valid_actions,throw):#Greedy bot which simulates every throw once and chooses the best expectation -> if rethrowing brings possibly a better result the risk is taken
    option = np.argwhere(valid_actions)
    choice,points = -1,0
    game = copy.deepcopy(state)
    for opt in range(len(option)):
        if option[opt][0] == 0:
            for perm in all_permutations:
                c_game = copy.deepcopy(game.get_next_state(game,player,0,np.asarray(perm)))
                c = greedy_bot_action(c_game,player,game.get_valid_moves(c_game,player,throw+1),throw+1)
                c_game = game.get_next_state(c_game,player,c)
                p,t = game.get_points_and_terminated(c_game)
                if p[player]>points:
                    choice,points = c,p[player]
        else:
            c_game = copy.deepcopy(game.get_next_state(game,player,option[opt][0]))
            p,t = game.get_points_and_terminated(c_game)
            if p[player]>points:
                choice,points = option[opt][0],p[player]   
    return choice

x = np.array([[0,0,0,0]])
st = time.process_time()
for i in range(100):
    Yahtzee = yahtzee(4)
    game = Yahtzee.get_initial_state()
    game = Yahtzee.get_next_state(game,0,0,[1,1,1,1,1])
    player = 0
    throw = 0
    while not Yahtzee.get_points_and_terminated(game)[1]:
        v = Yahtzee.get_valid_moves(game,player,0)
        act = random_bot_action(Yahtzee,game,player,v)
        game = Yahtzee.get_next_state(game,player,act)
        game = Yahtzee.get_next_state(game,player,0,[1,1,1,1,1])
        player = (player+1)%4

    points, t = Yahtzee.get_points_and_terminated(game)
    x = np.append(x,[points],axis=0)
et = time.process_time()
res = et - st
print('CPU Execution time:', res, 'seconds')
x = x[1:]
print(np.max(x,axis=0))
print(np.min(x,axis=0))
print(np.average(x,axis=0))
print(np.median(x,axis=0))
