import numpy as np
from numpy import random
import random as r
import time
import itertools as iter
from tqdm import tqdm

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

#initiates a players turn with expexted input for rethrow and dice choice for the rethrow
def player_turn():
    d = dice_throw(5)
    print("Your dice: {}".format(d))
    while True:
        r = input("Throw again? Type y/n: ")
        if r == "y" or r == "n":
            break
        print("Not a valid choice. Please try again!")
    i=0
    while r=="y" and i < 2:
        to_rethrow = [False,False,False,False,False]
        x = input("Choose dice to rethrow in format x x x: ")
        a = [int(temp) for temp in x.split() if temp.isdigit()]
        for y in a:
            to_rethrow[y-1] = True 
        d = rethrow(d,to_rethrow)
        print("Your dice: {}".format(d))
        i += 1
        if i < 2:
            while True:
                r = input("Throw again? Type y/n: ")
                if r == "y" or r == "n":
                    break
                print("Not a valid choice. Please try again!") 
    return d 

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
    return [True,False] if cnt_longest_seq == 4 else [True,True] if cnt_longest_seq == 5 else [False,False] #length can only be 4 or 5

#all points can be set/adjusted here
count_eyes = lambda x, i: np.count_nonzero(x==i)*i
three_same = lambda x: np.sum(x) if any([np.count_nonzero(x==i)==3  for i in range(1,6)]) else 0
four_same = lambda x: np.sum(x) if any([np.count_nonzero(x==i)==4  for i in range(1,6)]) else 0
fullHouse = lambda x: 25 if full_house(x) else 0
small_straight = lambda x: 30 if straight(x)[0] else 0    
large_straight = lambda x: 40 if straight(x)[1] else 0
yahtzee = lambda x: 50 if np.count_nonzero(x==x[0]) == 5 else 0
chance = lambda x: np.sum(x)

#calculate total result
def calc_total_res(result):
    res = np.zeros(len(result))
    for i in range(len(result)):
        x = 35 if np.sum(result[i][0:6]) >= 63 else 0 #Bonus
        res[i] = np.sum(result[i]) + x
    
    return res

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
#a list for a players board
def result_board():
    return np.array([-1 for i in range(13)])

class yahtzee:
    #random bot action. returns Index of chosen option
    def random_bot(self,result,d):
        #RANDOM BOT     
        for i in range(2):
            d = rethrow(d,r.choices([True,False], k = 5))
        x = r.choice([i for i in range(len(result)) if result[i] == -1])
        result[x] = options[x](d)
        return x
    
    #greedy bot action. returns Index of chosen option
    def greedy_bot(self,result,d):
        for i in range(2):
            curr_points = np.array([])
            for opt in options:
                curr_points = np.append(curr_points,opt(d))
            curr_points[np.where(result!=-1)] = -1
            maximum_points = np.argmax(curr_points)
            if curr_points[maximum_points] != 0:
                break
            else:
                d = rethrow(d,[True,True,True,True,True])

        if curr_points[maximum_points] != 0:
            result[maximum_points] = options[maximum_points](d)
        else:#if 0 point have to be entered, select a random choice
            maximum_points = r.choice([i for i in range(len(result)) if result[i] == -1])
            result[maximum_points] = options[maximum_points](d)
        return maximum_points

    #bot turn. Gives only information about a turn if it contains human player
    def bot_turn(self,result,botID):
        if not self.isBotGame:
            print("\n Bot Turn")
        d = dice_throw(5)
        if botID == 1:
            x = self.random_bot(result,d)
        elif botID == 2:
            x = self.greedy_bot(result,d)
        else:
            raise Exception("Invalid BotID!")
        
        if not self.isBotGame:
            print("Player entered dice throw {} in {}".format(d,option_names[x]))
        return 0
    
    #Interactive turn for humane player with Texts and inputs
    #int:current_player, list:result -> current players point sheet
    def human_turn(self,current_player,result):
        print("\n Player {} Turn beginns:".format(current_player))
        dice = player_turn()
        print("Choose where to enter your results")
        print("--------------------------")
        for i in range(13):
            if result[i] == -1:
                print(option_names[i] + "  {}".format(i))
        print("--------------------------")
        choice = -1
        unmarked = [i for i in range(len(result)) if result[i] == -1]
        while choice not in unmarked:           
            try:
                choice = int(input("Your Choice: "))
            except:
                print("Choice must be a number!")
            
            if(choice == 100):
                raise Exception("Game exit chosen")
            if choice not in unmarked:
                print("Please Enter valid choice!") 
        result[choice] = options[choice](dice)

    #main method to simulate a game
    def gameplay(self):
        current_player = 0
        while np.any([-1 in self.res[i] for i in range(len(self.res))]):
            if self.playerIDs[current_player] == 0:
                self.human_turn(current_player,self.res[current_player])
            else:
                self.bot_turn(self.res[current_player],self.playerIDs[current_player])
            current_player = (current_player+1)%len(self.res)

    def print_results(self):
        #print results
        print("Final Results:")
        results = calc_total_res(self.res)
        for i in range(len(results)):
            print("Result player {} : {}".format(i, results[i]))

    #human players is a list of playerIDs [x,x,x,x]
    #0=human; 1=random Bot; 2=greedy Bot
    def __init__(self,num_players,playerIDs):
        assert(num_players>0 and num_players<5) and all(playerIDs[x] in np.arange(3) for x in range(num_players)) and len(playerIDs)==num_players
        self.res = [result_board() for i in range(num_players)]
        self.playerIDs = playerIDs
        self.isBotGame = 0 not in self.playerIDs
        self.gameplay()

#Test runtime of n games with x bots
#get the start time
st = time.process_time()
x = [calc_total_res(yahtzee(4,[2,2,2,2]).res)]
for i in tqdm(range(9999)):
    x = np.append(x,[calc_total_res(yahtzee(4,[2,2,2,2]).res)],axis=0)
# get the end time
et = time.process_time()
# get execution time
res = et - st
print('CPU Execution time:', res, 'seconds')
print(np.average(x,axis=0))
print(np.max(x,axis=0))
print(np.min(x,axis=0))
print(np.median(x,axis=0))

#Test winrate of each bot in 10k games
#get the start time
# st = time.process_time()
# x = [np.argmax(calc_total_res(yahtzee(2,[1,2]).res))]
# for i in range(9999):
#     x = np.append(x,np.argmax(calc_total_res(yahtzee(2,[1,2]).res)))
# # get the end time
# et = time.process_time()
# # get execution time
# res = et - st
# print('CPU Execution time:', res, 'seconds')
# cnt = np.count_nonzero(x)
# print("Total Games: {}, Wins random Bot: {}, Wins Greedy Bot: {}".format(len(x),len(x)-cnt,cnt))
# print("Win rate random Bot: {}%, Win rate Greedy Bot: {}%".format(100*((len(x)-cnt)/len(x)),100*(cnt/len(x))))


# r contains all possible permutations for 5 dice
# use to calculate all chances for the markable events (yahtzee,...) 
# r = list(iter.product(range(1,7),repeat=3))
# length  = len(r)
# print(12/length)
# cnt = {}
# for d in r:
#     val = options[11](np.asarray(d))
#     if val not in cnt:
#         cnt[val] = 0
#     cnt[val] += 1
# print(cnt)
# prob = 0
# for i in cnt:
#     if i == 0:
#         cnt[i] = cnt[i]/length
#     else:
#         prob += cnt[i]
# print(cnt[0])
# print(prob/length)

# cnt = {}
# for d in r:
#     val = np.sum(np.asarray(d))
#     if val not in cnt:
#         cnt[val] = 0
#     cnt[val] += 1
# print(cnt)
