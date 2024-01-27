# AlphaZero-Approach-for-dice-games
In this project multiple algorithms are used to create the best possible AI for Yahtzee, Quixx and Machi Koro, three dice games. The final goal is to create an AI which uses a similar approach as AlphaZero.

To achieve this multiple approaches have to be regarde such as MiniMax, MCTS and more. But since the dice games are neither deterministic nor 2-player games these ideas have to be reformed using chance nodes.
The idea for MiniMax is to expand with chance nodes to Expectiminimax and combining it with Max^n for >2 players. The result should be a "ExpectiMax^n" algorithm.
Since the MCTS with chance nodes would converged towards Expectimax this idea must only be explored to get the best basis to apply a NN on it. 

To reduce the branching factor the methods of pruning or open loop have to be adapted for the changed algorithms.

\
Yahtzee: large branching but straight forward actions (Maybe because of the large branching factor a simplified game is used e.g. less dice or instead of D6 use D4, possibly create a single player AI which only tries to maximize his own result)\
Quixx: in comparison low branching but complex turn actions (white dice turns) \
Machi Koro: Options per turn are mostly very low since many limitations are created with dice and bank value. Assuming an unlimited amount of coins for the current player there are max. 19 cards to buy + 1 choice of buying none. Tree depth can be reduced by adding a rule that players must buy something in a number of turns or the game end. E.g. assuming 4 players don't buy anything for two whole roatations the game end with every player as a loser. (Everyone loses so this state is not desiable)
MachiKoro balancing: in a 2 player game active stealing is same as take from everyone. So take 2 coins is buffed to take three and steal 5 is disabled

\
Git project for AlphaZero for TicTacToe and Connect4 as inspiration: https://github.com/foersterrobert/AlphaZeroFromScratch
