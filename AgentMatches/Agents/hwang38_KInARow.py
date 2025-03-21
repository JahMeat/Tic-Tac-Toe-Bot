'''
hwang38_KInARow.py
Authors: Wang, Heidi

An agent for playing "K-in-a-Row with Forbidden Squares" and related games.
CSE 415, University of Washington

'''

from agent_base import KAgent
from game_types import State, Game_Type
import game_types as game_types

AUTHORS = 'Heidi Wang' 

import time # You'll probably need this to avoid losing a
 # game due to exceeding a time limit.

# Create your own type of agent by subclassing KAgent:

GAME_TYPE = None
CURR_PLAYER = None
MAX, MIN = 'X', 'O'
FORBIDDEN = '-'
MAX_INT = 9223372036854775807
MIN_INT = -MAX_INT

class OurAgent(KAgent):  # Keep the class name "OurAgent" so a game master
    # knows how to instantiate your agent class.

    def __init__(self, twin=False):
        self.twin=twin
        self.nickname = 'MK'
        self.long_name = 'Mark Lee'
        self.persona = 'carefree'
        self.alpha_beta_cutoffs_this_turn = -1
        self.num_static_evals_this_turn = -1
        self.zobrist_table_num_entries_this_turn = -1
        self.zobrist_table_num_hits_this_turn = -1

    def introduce(self):
        intro = '\nMy name is Mark Lee.\n'+\
            'Heidi Wang (hwang38) made me.\n'+\
            'My name is Mark. You can mark me in your heart.'+\
            'Let\'s get it!\n'
        if self.twin: intro += "By the way, I'm the TWIN.\n"
        return intro

    def nickname(self): return self.nickname

    # Receive and acknowledge information about the game from
    # the game master:
    def prepare(
        self,
        game_type,
        what_side_to_play,
        opponent_nickname,
        expected_time_per_move = 0.1, # Time limits can be
                                      # changed mid-game by the game master.

        utterances_matter=True):      # If False, just return 'OK' for each utterance,
                                      # or something simple and quick to compute
                                      # and do not import any LLM or special APIs.
                                      # During the tournament, this will be False.

        # Optionally, import your LLM API here.
        # Then you can use it to help create utterances.
        if utterances_matter:
            global random
            import random

        # Write code to save the relevant information in variables
        # local to this instance of the agent.
        # Game-type info can be in global variables.
        self.who_i_play = what_side_to_play
        self.opponent_nickname = opponent_nickname
        self.time_limit = expected_time_per_move
        global GAME_TYPE
        GAME_TYPE = game_type
        self.current_game_type = GAME_TYPE
        self.utterances_matter = utterances_matter
        if self.twin: self.utt_count = 5 # Offset the twin's utterances.

        return "OK"

    # The core of your agent's ability should be implemented here:             
    def make_move(self, current_state, last_utterance, time_limit=1000,
                  autograding=False, use_alpha_beta=True,
                  use_zobrist_hashing=False, max_ply=3,
                  special_static_eval_fn=None):
        self.start_time = time.time()
        # print("make_move has been called")

        if not special_static_eval_fn:
            special_static_eval_fn = lambda state : self.static_eval(state)

        if use_alpha_beta:
            self.alpha_beta_cutoffs_this_turn = 0

        self.num_static_evals_this_turn = 0
        myMove = self.minimax(current_state, max_ply, time_limit, special_static_eval_fn, use_alpha_beta)
        # print(myMove)
        newVal, newMove, newState = myMove

        newRemark = self.nextUtterance(newVal)

        # print("Returning from make_move")
        if not autograding:
            return [[newMove, newState], newRemark]

        stats = [self.alpha_beta_cutoffs_this_turn,
                 self.num_static_evals_this_turn,
                 self.zobrist_table_num_entries_this_turn,
                 self.zobrist_table_num_hits_this_turn]

        return [[newMove, newState]+stats, newRemark]

    # The main adversarial search function:
    def minimax(self,
            state,
            depth_remaining,
            time_limit,
            static_eval_fn,
            pruning=False,
            alpha=None,
            beta=None):

        if depth_remaining == 0:
            self.num_static_evals_this_turn += 1
            return [static_eval_fn(state), None, state]

        whoseMove = state.whose_move
        provisional = MIN_INT if whoseMove == MAX else MAX_INT
        if pruning:
            if not alpha:
                alpha = MIN_INT
            if not beta:
                beta = MAX_INT

        chosen_move = None
        chosen_state = None

        successors, moves = successors_and_moves(state)
        if not successors:
            self.num_static_evals_this_turn += 1
            return [static_eval_fn(state), None, state]
        for i,s in enumerate(successors):
            newState = State(s, other(whoseMove))
            if time.time() > self.start_time + time_limit - 1:
                newVal = self.minimax(newState, 0, 
                                      time_limit, static_eval_fn, pruning, alpha, beta)
            else:
                newVal = self.minimax(newState, depth_remaining -1, 
                                      time_limit, static_eval_fn, pruning, alpha, beta)
            newVal = newVal[0]
            if ((whoseMove == MAX and newVal > provisional) 
                or (whoseMove == MIN and newVal < provisional)):
                provisional = newVal
                chosen_state = s
                chosen_move = moves[i]
            if pruning:
                if whoseMove == MAX:
                    alpha = max(newVal, alpha)
                if whoseMove == MIN:
                    beta = min(newVal, beta)
                if alpha >= beta:
                    break
        
        return [provisional, chosen_move, chosen_state, ]
 
    # Values should be higher when the states are better for X,
    # lower when better for O.
    def static_eval(self, state, game_type=None):
        if not game_type:
            game_type = GAME_TYPE

        k = game_type.k
        board = state.board

        val = 0
        # print(state)
        x, o = find_val(k, board)
        for i in range(1,len(x)):
            val += 10**(i-1) * x[i]
            val -= 10**(i-1) * o[i]
        # print(val)
        return val
    
    # Uses IF-THEN conditionals to trigger utterance patterns 
    # based on how long the game has been going, 
    # quantified by UTTERANCE_COUNT, 
    # and on whether a move the agent makes has a value 
    # that is good or bad for it. 
    # A long game only has a 0.2 chance of affecting the utterance 
    # when UTTERANCE_COUNT > 10. 
    # Once the type of utterance is decided, 
    # the particular utterance is chosen randomly among 
    # UTTERANCE_BANKs for each type. 
    # Utterances will not repeat until all unique utterances 
    # of that type have been used up.
    def nextUtterance(self, newVal):
        if not self.utterances_matter:
            return "OK"
        if ((newVal <= 0 and self.who_i_play == MAX) or
              (newVal > 0 and self.who_i_play == MIN)):
            utt_bank = UTTERANCE_BANK_NEG
            used_bank = USED_NEG
        else:
            utt_bank = UTTERANCE_BANK_POS
            used_bank = USED_POS
        if UTTERANCE_COUNT > 10 and random.random() < 0.2:
            utt_bank = UTTERANCE_BANK_LONG
            used_bank = USED_LONG
        if not utt_bank:
            while used_bank:
                utt_bank.append(used_bank.pop())
        r = int(random.random() * len(utt_bank))
        print(r)
        utt = utt_bank.pop(r)
        used_bank.append(utt)
        # return "[" + str(newVal) + "] " + utt
        return utt
 
# OPTIONAL THINGS TO KEEP TRACK OF:

#  WHO_MY_OPPONENT_PLAYS = other(WHO_I_PLAY)
#  MY_PAST_UTTERANCES = []
USED_NEG = []
USED_POS = []
USED_LONG = []
#  OPPONENT_PAST_UTTERANCES = []
UTTERANCE_COUNT = 0
#  REPEAT_COUNT = 0 or a table of these if you are reusing different utterances


# FOR EACH OF THE FOLLOWING FOUR FUNCTIONS:
# Given an int k valid by the constraints of a game,
# a board valid by the constraints of a game,
# a k-size array for MAX, and a k-size array for MIN,
# for each index i from 1 to k,
# adds the occurrences of i x's or o's in a line to x or o, respectively,
# and returns the number of k-length lines considered.

# a line is a row, column,
# diagonal going downwards left to right,
# or diagonal going upwards left to right
# for each of the following four functions, in order.

# what makes n/k in a line:
# line of k cells
# n X in the line
# no O in the line
def rows(x, o, k, board):
    count = 0
    for i in range(len(board)):
        for j in range(len(board[0])-k+1):
            countX = 0
            countO = 0
            blocked = False
            for w in range(k):
                cell = board[i][j+w]
                if cell == FORBIDDEN:
                    blocked = True
                    break
                if cell == MAX:
                    countX += 1
                if cell == MIN:
                    countO += 1
            if not blocked:
                if countX == 0 and countO != 0:
                    o[countO] += 1
                if countO == 0 and countX != 0:
                    x[countX] += 1
                count += 1
    return count

# see nearest comment above
def cols(x, o, k, board):
    count = 0
    for i in range(len(board)-k+1):
        for j in range(len(board[0])):
            countX = 0
            countO = 0
            blocked = False
            for w in range(k):
                cell = board[i+w][j]
                # print("cell", i+w, j, cell)
                if cell == FORBIDDEN:
                    blocked = True
                    break
                if cell == MAX:
                    countX += 1
                if cell == MIN:
                    countO += 1
            if not blocked:
                # print(countX, countO)
                if countX == 0 and countO != 0:
                    o[countO] += 1
                if countO == 0 and countX != 0:
                    x[countX] += 1
                count += 1
    return count

# see nearest comment above
def ddiags(x, o, k, board):
    count = 0
    for i in range(len(board)-k+1):
        for j in range(len(board[0])-k+1):
            countX = 0
            countO = 0
            blocked = False
            for w in range(k):
                cell = board[i+w][j+w]
                if cell == FORBIDDEN:
                    blocked = True
                    break
                if cell == MAX:
                    countX += 1
                if cell == MIN:
                    countO += 1
            if not blocked:
                if countX == 0 and countO != 0:
                    o[countO] += 1
                if countO == 0 and countX != 0:
                    x[countX] += 1
                count += 1
    return count

# see nearest comment above
def udiags(x, o, k, board):
    count = 0
    for i in range(len(board)-k+1):
        for j in range(len(board[0])-k+1):
            countX = 0
            countO = 0
            blocked = False
            for w in range(k):
                cell = board[len(board)-1-(i+w)][j+w]
                if cell == FORBIDDEN:
                    blocked = True
                    break
                if cell == MAX:
                    countX += 1
                if cell == MIN:
                    countO += 1
            if not blocked:
                if countX == 0 and countO != 0:
                    o[countO] += 1
                if countO == 0 and countX != 0:
                    x[countX] += 1
                count += 1
    return count

# Given an int k valid by the constraints of a game
# and a board valid by the constraints of a game,
# returns a two-tuple containing 
# a k-size array x for MAX and a k-size array o for MIN,
# where for each index i from 1 to k,
# the value at that index is the number of occurrences 
# of i x's or o's in a line in array x or o, respectively.
# Calls the four functions for each of the four types of lines.
def find_val(k, board):
    x = [0] * (k+1)
    o = [0] * (k+1)
    count = rows(x, o, k, board)
    # print(x, count)
    # print(o, count)
    # x = [0] * (k+1)
    # o = [0] * (k+1)
    count = cols(x, o, k, board)
    # print(x, count)
    # print(o, count)
    count = ddiags(x, o, k, board)
    count = udiags(x, o, k, board)
    return (x, o)


# Figure out who the other player is.
# For example, other("X") = "O".
def other(p):
    if p=='X': return 'O'
    return 'X'

# The following is a Python "generator" function that creates an
# iterator to provide one move and new state at a time.
# It could be used in a smarter agent to only generate SOME of
# of the possible moves, especially if an alpha cutoff or beta
# cutoff determines that no more moves from this state are needed.
def move_gen(state):
    b = state.board
    p = state.whose_move
    o = other(p)
    mCols = len(b[0])
    nRows = len(b)

    for i in range(nRows):
        for j in range(mCols):
            if b[i][j] != ' ': continue
            new_s = do_move(state, i, j, o)
            yield [(i, j), new_s]

# This uses the generator to get all the successors.
def successors_and_moves(state):
    moves = []
    new_states = []
    for item in move_gen(state):
        moves.append(item[0])
        new_states.append(item[1])
    return [new_states, moves]

# Perform a move to get a new state.
def do_move(state, i, j, o):
            new_s = State(old=state)
            new_s.board[i][j] = state.whose_move
            new_s.whose_move = o
            return new_s

# 5
UTTERANCE_BANK_NEG = ["... Well, I feel like the possibility of all those possibilities being \n"+\
                          "possible is just another possibility that can possibly happen...",
                      "It really isn't. It really really isn't though. Like it really really isn't.",
                      "You think you're winning, but winners are those who never give up.",
                      "Even if I'm losing, I got my back up.",
                      "This is for you!... \n"+\
                          "Haha,ha... That didn't just happen..."
                      ]

# 7
UTTERANCE_BANK_POS = ["I know just where to be.", 
                      "Don't ever try to come even closer.",
                      "Five Guys, goodbye guys.",
                      "We are at the tram stop. Are you the tram that stops? \n"+\
                          "I'm gonna stop, you in your tracks, ...",
                      "You gotta seize the opportunity.",
                      "Ayo, listen up! We gon resonate.",
                      "Hater hater talk talk."
                      ]

# 3
UTTERANCE_BANK_LONG = ["We've been playing for so long, but winners are those who never give up.", 
                       "This a long ah ride.",
                       "This game feels longer than my schedules."
                      ]


def test():
    global GAME_TYPE
    GAME_TYPE = game_types.TTT
    print(GAME_TYPE)
    h = OurAgent()
    print("I am ", h.nickname)
    
    ttt = GAME_TYPE.initial_state
    print("ttt initial state: ")
    print(ttt)
    print("successors_and_moves: ")
    print(successors_and_moves(ttt))

if __name__=="__main__":
    test()