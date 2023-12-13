# AlphaZero-Approach-for-dice-games
In this project multiple algorithms are used to create the best possible AI for Yahtzee, Quixx and Machi Koro, three dice games. The final goal is to create an AI which uses a similar approach as AlphaZero.

To achieve this multiple approaches have to be regarde such as MiniMax, MCTS and more. But since the dice games are neither deterministic nor 2-player games these ideas have to be reformed using chance nodes.
The idea for MiniMax is to expand with chance nodes to Expectiminimax and combining it with Max^n for >2 players. The result should be a "ExpectiMax^n" algorithm.
Since the MCTS with chance nodes would converged towards Expectimax this idea must only be explored to get the best basis to apply a NN on it. 

To reduce the branching factor the methods of pruning or open loop have to be adapted for the changed algorithms.

\
Yahtzee: large branching but straight forward actions (Maybe because of the large branching factor a simplified game is used e.g. less dice or instead of D6 use D4, possibly create a single player AI which only tries to maximize his own result)\
Quixx: in comparison low branching but complex turn actions (white dice turns) \
Machi Koro: TBD
