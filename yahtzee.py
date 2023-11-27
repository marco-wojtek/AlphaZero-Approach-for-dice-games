import numpy as np
from numpy import random
import random as r

#returns a numpy array with num_dice values between 1 and 6 both included
def dice_throw(num_dice=1):
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

    if len(sorted_dice) < 4:
        return [False,False]
    for n in range(1,length):
        if sorted_dice[n] != sorted_dice[n-1]+1:
            return [False,False]
    return [True,False] if length == 4 else [True,True] #length can only be 4 or 5

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
    assert len(result) == 13
    x = 35 if np.sum(result[0:6]) >= 63 else 0 #Bonus
    return np.sum(result) + x

option_names = np.array(["Ones",
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

options = np.array([lambda x: count_eyes(x,i=1),
           lambda x: count_eyes(x,i=2),
           lambda x: count_eyes(x,i=3),
           lambda x: count_eyes(x,i=4),
           lambda x: count_eyes(x,i=5),
           lambda x: count_eyes(x,i=6),
           three_same,
           four_same,
           fullHouse,
           small_straight,
           large_straight,
           yahtzee,
           chance])
#a list for a players board
def result_board():
    return np.array([-1 for i in range(13)])

class yahtzee:
    #bot turn. Gives only information about a turn if it contains human player
    def bot_turn(self,result,bot_game=False):
        if not bot_game:
            print("\n Bot Turn")
        d = dice_throw(5)       
        for i in range(2):
            d = rethrow(d,r.choices([True,False], k = 5))
        x = r.choice([i for i in range(len(result)) if result[i] == -1])
        result[x] = options[x](d)

        if not bot_game:
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
            if self.human_players[current_player]:
                self.human_turn(current_player,self.res[current_player])
            else:
                self.bot_turn(self.res[current_player],True not in self.human_players)
            current_player = (current_player+1)%len(self.res)
        
        print("Final Results:")
        for i in range(len(self.res)):
            print("Result player {} : {}".format(i, calc_total_res(self.res[i])))

    def __init__(self,num_players,human_players):
        assert(num_players>0 and num_players<5)
        self.res = [result_board() for i in range(num_players)]
        self.human_players = human_players
        self.gameplay()
    
x = yahtzee(2,[False,True])
print(x.res)

#TODO:
#-improve file/class/method structure

