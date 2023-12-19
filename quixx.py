import numpy as np
from numpy import random
import time
import math

colour_dict = { 0 : "White ",
                1 : "Red   ",
                2 : "Yellow",
                3 : "Green ",
                4 : "Blue  ",
                5 : "Error "}

class player_sheet:

    def print_sheet(self):
        for i in range(1,len(self.sheet)+1):
            print("{}: {}".format(colour_dict[i],self.sheet[i]))
    #method receives the number to 'mark' and the 'row' in which the mark should be placed
    # row in {1,2,3,4,5}, mark in {2,3,4,5,6,7,8,9,10,11,12}
    # returns error code 0 if no exception was found
    def enter_throw(self,row,mark,closed_rows):
        try:
            if row == 5:
                self.sheet[row] +=1
            elif closed_rows[row-1]:
                return -1
            else:
                index = np.where(self.sheet[row] == mark)[0][0]
                s = np.where(self.sheet[row] == 1)[0]
                start = s[-1] +1 if len(s) > 0 else 0
                #the last value of a row cannot be marked if there are not already 5 marks in that row
                if index==len(self.sheet[row])-1 and np.count_nonzero(self.sheet[row]==1) < 5:
                    return -1
                self.sheet[row][start:index] = 0
                self.sheet[row][index] = 1
                #if the last value is marked the row is closed and the player get a bonus mark as an additional 1 added to the array
                if index==len(self.sheet[row])-1 and np.count_nonzero(self.sheet[row]==1) >= 5:
                    self.sheet[row] = np.append(self.sheet[row],1)
            return 0
        except:
            return -1

    def __init__(self):
        self.sheet = {
            1 : np.arange(2,13),
            2 : np.arange(2,13),
            3 : np.arange(12,1,-1),
            4 : np.arange(12,1,-1),
            5 : np.array(0) #error Block; if over the value 3 the game ends              
        }

class dice:
    #Colours based on int val   
    def throw_dice(self):
        self.eye = random.randint(1,7)

    def dice_val(self):
        return [self.eye, self.colour]
    
    def __init__(self,c=0):
        self.colour = c
        self.eye = 0

class quixx:

    def throw_dice(self):
        for d in self.dices:
            d.throw_dice()
    
    #returns an array which contains all possible inputs for each valid combination; doesn't filter unusable options
    def calc_options(self,current_player):
        opt = np.array([[0,0],
                       [0,0],
                       [0,0],
                       [0,0],
                       [0,0]]) #opt array represents all the possible combinations, first row is the sum of both white dice, the other rows are the two possibilities for each colured dice with the white dice
        opt[:,0] = self.dices[0].dice_val()[0]
        opt[:,1] = self.dices[1].dice_val()[0]
        #add coloured dice values to the respective row
        for i in range(1,5):
            opt[i,:] += self.dices[i+1].dice_val()[0]
            s = self.sheets[current_player].sheet[i]
            if len(np.where(s == opt[i,0])[0])==0:
                opt[i,0] = 0
            if len(np.where(s == opt[i,1])[0])==0:
                opt[i,1] = 0
                
        #adjust the white dice values
        opt[0,0] += self.dices[1].dice_val()[0]
        opt[0,1] = opt[0,0]
        #closed rows are no options anymore thus the options are set to 0
        opt[np.where(np.append([False],self.closed_rows)),:] = 0
        return opt
    
    def print_opt(self,opt):
        print("Possible Options")
        names = np.array(["Red","Yellow","Green","Blue"])
        for i in range(len(names)):
            n = opt[i+1,np.where(opt[i+1]!=0)[0]]
            if len(n) == 2:
                print("Number {} or {} in row {}".format(n[0],n[1],names[i]))
            elif len(n) == 1:
                print("Number {} in row {}".format(n[0],names[i]))
            else:
                print("No number in row {}".format(names[i]))          
    
    #the result is the sum of the points of each row minus 5 x the number of marks in the error row
    #the point for one row is dependent on the number of marks (see quixx_player_sheet.png) 
    def calc_result(self):
        points = np.array([0,1,3,6,10,15,21,28,36,45,55,66,78])
        sum = np.zeros(self.num_players)
        for s in range(len(self.sheets)):
            var = self.sheets[s]
            for r in range(1,5):
                sum[s] += points[np.sum(var.sheet[r])]
            sum[s] -= 5 * var.sheet[5]
        return sum
    
    def white_dice_rotation(self,current_player):
        curr_player = current_player
        r = False # value to return; shows wether the current player entered the white dice
        #calc sum of the white dice
        white_dice_val = self.dices[0].dice_val()[0] + self.dices[1].dice_val()[0]
        while True:
            
            if self.playerIDs[curr_player] == 0:               
                #human player
                print("Player {} to choose!".format(curr_player))
                vals = [int(i) for i in input("Select Row to enter Value {} or enter nothing: ".format(white_dice_val)).split() if i.isdigit()]
                while len(vals) != 1 or self.sheets[curr_player].enter_throw(vals[0],white_dice_val,self.closed_rows) == -1:
                    #if no input is given the loop is skipped
                    if len(vals) == 0:
                        break
                    print("Invalid Option! Please try again!")
                    vals = [int(i) for i in input("Select Row to enter Value {} or enter nothing: ".format(white_dice_val)).split() if i.isdigit()]
                if len(vals) == 1 and curr_player==current_player:
                    r=True                    
            elif self.playerIDs[curr_player] ==1:
                #bot_player random          
                # selects a random row to enter the white dice value; if not successful due to invalid option or other. white dice are ignored    
                if self.sheets[curr_player].enter_throw(random.randint(1,5),white_dice_val,self.closed_rows) != -1 and curr_player==current_player:
                    r=True
            elif self.playerIDs[curr_player] in [2,3]:
                row_r_y = np.arange(2,13)
                row_g_b = np.arange(12,1,-1)
                last_marked = np.zeros(4)
                for i in range(1,5):
                    #get index of last marked number
                    try:
                        index = np.where(self.sheets[current_player].sheet[i] == 1)[0][0]
                        last_marked[i-1] = row_r_y[index] if i in [1,2] else row_g_b[index]
                    except:
                        # if none was marked use following values
                        last_marked[i-1] = 1 if i in [1,2] else 13
                    if (i in [1,2] and last_marked[i-1] > white_dice_val) or (i in [3,4] and last_marked[i-1] < white_dice_val):
                        last_marked[i-1] = np.inf
                    elif (i in [1,2] and last_marked[i-1] < white_dice_val):
                        last_marked[i-1] = white_dice_val - last_marked[i-1]
                    elif (i in [3,4] and last_marked[i-1] > white_dice_val):
                        last_marked[i-1] -= white_dice_val
                best_rows = np.argsort(last_marked)
                
                for i in range(len(last_marked)):
                    #limiter for Greedy bot; if the white dice would leave out more than 3 numbers the white dice should be ignored
                    if self.playerIDs[curr_player] == 3 and last_marked[best_rows[0]] > 3:
                        break
                    if self.sheets[curr_player].enter_throw(best_rows[i],white_dice_val,self.closed_rows) != -1 and curr_player==current_player:
                        r=True
                        break
                
            #one roataion around all players
            curr_player = (curr_player+1)%self.num_players
            if curr_player == current_player:
                break
        return r
    
    def random_bot(self,current_player,options):
        #RANDOM BOT
        opt = np.arange(1,5)
        random.shuffle(opt) #shuffled vals
        for i in range(len(opt)):
            val = opt[i]

            if options[val,0] != 0 and (self.sheets[current_player].enter_throw(val,options[val,0],self.closed_rows) != -1 or self.sheets[current_player].enter_throw(val,options[val,1],self.closed_rows) != -1):
                return True
        return False
    
    def greedy_bot(self,current_player,options):
        #GREEDY-LIKE BOT: tries to make as much marks as possible and marks in the row where the new value has the smallest distance to the last marked
        #maske all values which are zero or white dice
        options_masked = np.ma.masked_equal(np.ma.masked_array(options,mask=[1,1,0,0,0,0,0,0,0,0]),0)
        row_r_y = np.arange(2,13)
        row_g_b = np.arange(12,1,-1)
        last_marked = np.zeros(4)
        #count unmasked values
        cnt = np.ma.count(options_masked)
        #if all values are marked no option is calculated thus no option can be selected
        if cnt != 0:
            for i in range(1,5):
                #get index of last marked number
                try:
                    index = np.where(self.sheets[current_player].sheet[i] == 1)[0][0]
                    last_marked[i-1] = row_r_y[index] if i in [1,2] else row_g_b[index]
                except:
                    # if none was marked use following values
                    last_marked[i-1] = 1 if i in [1,2] else 13
            #calculate the distances between options and last marked
            for i in range(1,len(options_masked)):
                options_masked[i] = last_marked[i-1] - options_masked[i]
            options_masked[1:3] = -options_masked[1:3]

            #get best option and enter throw
            best_option = np.dstack(np.unravel_index(np.argmin(np.ravel(options_masked)),options_masked.shape))[0][0]
            row = best_option[0]
            number = options[row,best_option[1]]
            self.sheets[current_player].enter_throw(row,number,self.closed_rows)
            return True
        return False

    def bot_turn(self,current_player):
        #Bot first randomly decides to enter the white dice or not
        #after that randomly iterate through the choices and test if they can be entered
        #if in both phases no value was marked select error row
        
        #print("Player {} (Bot) turn".format(current_player))

        self.throw_dice()

        entered_throw = self.white_dice_rotation(current_player)

        options = self.calc_options(current_player)
        
        if self.playerIDs[current_player] == 1:
            entered_throw = self.random_bot(current_player,options) or entered_throw
        elif self.playerIDs[current_player] in [2,3]:
            entered_throw = self.greedy_bot(current_player,options) or entered_throw
        else:
            raise Exception("Invalid BotID!")
        
        if not entered_throw:
            self.sheets[current_player].enter_throw(5,0,self.closed_rows)
            
        if not self.isBotGame:
            print("Player {} (Bot) sheet after the turn".format(current_player))
            self.sheets[current_player].print_sheet()
        return 0
    
    #human player turn
    def player_turn(self,current_player):
        print("Player {} turn".format(current_player))
        self.throw_dice()        
        for d in self.dices:
            val, colour = d.dice_val()
            print(val, colour_dict[colour])

        #method for white dice rotation
        white_choice = self.white_dice_rotation(current_player)
        #current players option selection for coloured dice  
        #calculate options    
        options = self.calc_options(current_player)
        self.print_opt(options)
        self.sheets[current_player].print_sheet()
        vals = [int(i) for i in input("Select Option as Row Number or nothing: ").split() if i.isdigit()]
        #if neither white nor coloured dice were chosen the throw will be entered in the error row
        if len(vals) == 0 and not white_choice:
            self.sheets[current_player].enter_throw(5,0,self.closed_rows) #mark is set to 0 
            return 0
        elif len(vals) == 0:
            return 0
        while len(vals) != 2 or (vals[1] not in options[vals[0]]) or self.sheets[current_player].enter_throw(vals[0],vals[1],self.closed_rows) == -1:
            print("Invalid Option! Please try again!")
            vals = [int(i) for i in input("Select Option as Row Number: ").split() if i.isdigit()]
            if len(vals) == 0 and not white_choice:
                self.sheets[current_player].enter_throw(5,0,self.closed_rows) #mark is set to 0 
                break
            elif len(vals) == 0:
                break
        return 0
    
    #returns true if the gamestate shows that the ended by the rules
    def end_turn(self):
        f = False
        for s in self.sheets:
            for i in range(1,5):
                self.closed_rows[i-1] = len(s.sheet[i]) == 12 #if any row has 12 Elements it has been closed
            f = f or (s.sheet[5]>=4) # if any player has 4 error marks the game is guaranteed to end
        return f or np.sum(self.closed_rows)>=2
    
    def gameplay(self):
        current_player = 0
        finished = False 
        while not finished:
            if self.playerIDs[current_player]==0:
                self.player_turn(current_player)
            else:
                self.bot_turn(current_player)
            #end_turn method which checks for closed rows
            finished = self.end_turn()
            current_player = (current_player +1) % self.num_players
        
        #game is finished so the unmarked spaces must be cleared
        #every value over 1 is an unmarked space which is then removed by setting it to zero
        for s in self.sheets:
            for i in range(1,5):
                s.sheet[i][s.sheet[i]>1] = 0
            
    def print_results(self):
        #print results
        for i in range(self.num_players):
            print("Player {} (is Bot = {} )has {} points: ".format(i,self.playerIDs[i] in [2,3,4],self.calc_result()[i]))
            print("Player {} sheet".format(i))
            self.sheets[i].print_sheet()

    def __init__(self,num_players,playerIDs):
        assert num_players in [2,3,4] and num_players == len(playerIDs) and all(playerIDs[x] in np.arange(4) for x in range(num_players))
        self.dices = [dice(0),dice(0),dice(1),dice(2),dice(3),dice(4)]
        self.sheets = [player_sheet() for i in range(num_players)]
        self.playerIDs = playerIDs
        self.isBotGame = 0 not in self.playerIDs
        self.closed_rows = [False,False,False,False] # the state of the rows has to be given in each attempt to enter results
        self.num_players = num_players
        self.gameplay()

#Test runtime of n games with x bots
# get the start time
# st = time.process_time()
# y = quixx(4,[3,3,3,3])
# x = [y.calc_result()]
# error_rows = np.array([])
# error_rows = np.append(error_rows,np.count_nonzero(y.closed_rows))
# for i in range(9999):
#     y = quixx(4,[3,3,3,3])
#     x = np.append(x,[y.calc_result()],axis=0)
#     error_rows = np.append(error_rows,np.count_nonzero(y.closed_rows))
# # get the end time
# et = time.process_time()
# # get execution time
# res = et - st
# print('CPU Execution time:', res, 'seconds')
# print(np.max(x,axis=0))
# print(np.min(x,axis=0))
# print(np.average(x,axis=0))
# print(np.median(x,axis=0))
# print(np.average(error_rows))
# print(np.count_nonzero(np.where(error_rows==0)))
# print(np.count_nonzero(np.where(error_rows==1)))
# print(np.count_nonzero(np.where(error_rows==2)))
# print(np.count_nonzero(np.where(error_rows==3)))
# print(np.count_nonzero(np.where(error_rows==4)))

#Test win rates for different Bots
# get the start time
# st = time.process_time()
# y = quixx(3,[2,1,3])
# x = [np.argmax(y.calc_result())]
# error_rows = np.array([])
# error_rows = np.append(error_rows,np.count_nonzero(y.closed_rows))
# for i in range(10000):
#     y = quixx(3,[2,1,3])
#     x = np.append(x,[np.argmax(y.calc_result())])
#     error_rows = np.append(error_rows,np.count_nonzero(y.closed_rows))
# # get the end time
# et = time.process_time()
# # get execution time
# res = et - st
# print('CPU Execution time:', res, 'seconds')
# cnt_1 = (np.count_nonzero(np.where(x==0))/len(x))*100
# cnt_2 = (np.count_nonzero(np.where(x==1))/len(x))*100
# cnt_3 = (np.count_nonzero(np.where(x==2))/len(x))*100
# print("After {} games the Bots have the following winrates:\n Random Bot: {}% \n Greedy Bot: {}% \n limited Greedy Bot: {}%".format(len(x),cnt_3,cnt_2,cnt_1))
#  Random Bot: 14.2985701429857%
#  Greedy Bot: 42.435756424357564%
#  limited Greedy Bot: 43.255674432556745%
#TODO: Test in 1v1s and 1v3s, ...

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
            arr.copy()],dtype=object)
    elif player_num == 3:
        return np.array([
            np.zeros(6), 
            arr.copy(),
            arr.copy(),
            arr.copy()],dtype=object)
    else:
        return np.array([
            np.zeros(6), 
            arr.copy(),
            arr.copy(),
            arr.copy(),
            arr.copy()],dtype=object)

print(get_initial_state(0,4))
# def get_valid_moves(self,state,white_roatation=False):
#     if white_roatation:
#         return state[0][0]+state[0][1]
#     else:
#         valid = np.zeros((4,2))    
#         valid[:,0] += state[0][0] 
#         valid[:,1] += state[0][1]
#         valid[0,:] += state[0][2]
#         valid[1,:] += state[0][3]
#         valid[2,:] += state[0][4]
#         valid[3,:] += state[0][5]
