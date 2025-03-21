'''
cadetank_KInARow.py
Authors: Tanaka, Cade

An agent for playing "K-in-a-Row with Forbidden Squares" and related games.
CSE 415, University of Washington

THIS IS A TEMPLATE WITH STUBS FOR THE REQUIRED FUNCTIONS.
YOU CAN ADD WHATEVER ADDITIONAL FUNCTIONS YOU NEED IN ORDER
TO PROVIDE A GOOD STRUCTURE FOR YOUR IMPLEMENTATION.

'''

from agent_base import KAgent
from game_types import State, Game_Type
import game_types as game_types
import random

AUTHORS = 'Cade Tanaka' 

import time # You'll probably need this to avoid losing a
 # game due to exceeding a time limit.

# Create your own type of agent by subclassing KAgent:

class OurAgent(KAgent):  # Keep the class name "OurAgent" so a game master
    # knows how to instantiate your agent class.

    def __init__(self, twin=False):
        self.twin=twin
        self.nickname = 'Kevin'
        if twin: self.nickname = 'Bob'
        self.long_name = 'Kevin the minion'
        if twin: self.long_name = 'Bob the minion'
        self.persona = 'bland'
        self.voice_info = {'Chrome': 10, 'Firefox': 2, 'other': 0}
        self.playing = "don't know yet" # e.g., "X" or "O".
        self.alpha_beta_cutoffs_this_turn = -1
        self.num_static_evals_this_turn = -1
        self.zobrist_table_num_entries_this_turn = -1
        self.zobrist_table_num_hits_this_turn = -1
        self.current_game_type = None

    def introduce(self):
        intro  = '\n Bello \n'+\
            '"Cade Tanaka, cadetank.\n'+\
            'Me like banana \n' +\
            f'Sometimes our minion forgets to talk in English. Im a program that will translate what hes thinking'

        return intro

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
                                      # During the tournament, this will be False..
       if utterances_matter:
           pass
           # Optionally, import your LLM API here.
           # Then you can use it to help create utterances.
           
       # Write code to save the relevant information in variables
       # local to this instance of the agent.
       # Game-type info can be in global variables.
       self.who_i_play = what_side_to_play
       self.opponent_nickname = opponent_nickname
       self.time_limit = expected_time_per_move
       global GAME_TYPE
       GAME_TYPE = game_type
       print("Bello I will play you in ", game_type.long_name)
       self.my_past_utterances = []
       self.opponent_past_utterances = []
       self.repeat_count = 0
       self.utt_count = 0

       return "OK"

    def nickname(self): return self.nickname
   
    # The core of your agent's ability should be implemented here:             
    def make_move(self, current_state, current_remark, time_limit = None,
                  autograding=False, use_alpha_beta=True,
                  use_zobrist_hashing=False, max_ply=3,
                  special_static_eval_fn=None):

        start_time = time.time()
        
        default = minimax(self, current_state, max_ply,
                          use_alpha_beta, -100000, 100000, time_limit, autograding, special_static_eval_fn)

        move, score = default
        
        prev_eval = self.static_eval(current_state, GAME_TYPE)
        
        new_state = do_move(current_state, move[0], move[1], other(current_state.whose_move))

        new_eval = self.static_eval(new_state, GAME_TYPE)

        stats = [self.alpha_beta_cutoffs_this_turn,
                 self.num_static_evals_this_turn,
                 self.zobrist_table_num_entries_this_turn,
                 self.zobrist_table_num_hits_this_turn]

        random_int = random.randint(1,4)
        
        if random_int == 1:
            new_remark = stats_based_remark(self,stats)
        elif random_int == 2:
            new_remark = score_based_remark(self, new_eval)
        elif random_int == 3:
            new_remark = random_remark(self)
        else:
            new_remark = score_differential(self, new_eval, prev_eval)

        self.my_past_utterances.append(new_remark)
        

        return [[move, new_state], new_remark]

    def static_eval(self, state, game_type=None):
        board = state.board
        x_tot_score = 0
        o_tot_score = 0
        tot_score = 0

        if game_type is not None:
            k = game_type.k
        else:
            k = GAME_TYPE
            
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == 'O':
                    o_tot_score += check_directions(self, i, j, board, 'O', k)
                elif board[i][j] == 'X':
                    x_tot_score += check_directions(self, i, j, board, 'X', k)
                    
        tot_score = (x_tot_score - o_tot_score)

        return tot_score

def stats_based_remark(self, stats):
    random_int = random.randint(1,2)
    if self.twin:
        name = "Bob"
    else:
        name = "Kevin"
        
    if random_int == 1:
        new_remark = f"{name} trimmed {stats[0]} states!"
    else:
        new_remark = f"{name} is tired after doing {stats[1]} static evaluations"
    

    return new_remark
    

def score_based_remark(self, new_eval):
    new_remark = ''
    if self.twin:
        name = "Bob"
    else:
        name = "Kevin"
    
    if new_eval == 0:
        new_remark = f"{name} is bored. The game state is too even"
        
    elif self.who_i_play == 'X':
        if new_eval > 50 :
            new_remark = random_score_remark(self)
        elif new_eval > 500:
            new_remark = f"{name} is excited! He has an advantage!"
        elif new_eval > 800:
            new_remark = f"{name} can smell a win"
            
    elif self.who_i_play == 'O':
        if new_eval < -50:
            new_remark = random_score_remark(self)
        elif new_eval < -400:
            new_remark = f"{name} has the advantage now"
        elif new_eval < -800:
            new_remark = f"{name} can sense the end is near for you"
            
    elif self.who_i_play == 'X':
        if new_eval < -50:
            new_remark = f"v is sad. He thinks he's losing"
        elif new_eval < -400:
            new_remark = f"Poor {name}. He's definitely losing now"
        elif new_eval < -800:
            new_remark = f"{name} wants you to go easy on him!"

    elif self.who_i_play == 'O':
        if new_eval < 50:
            new_remark = f"{name} thinks he's losing"
        elif new_eval > 500:
            new_remark = f"{name}'s defintely losing now"
        elif new_eval > 800:
            new_remark = f"{name} wants you to go easy on him!"

    if new_remark == '':
        new_remark = random_remark(self)
    
    return new_remark

def score_differential(self, new_eval, prev_eval):
    diff = prev_eval - new_eval

    opp = diff * -1

    new_remark = ""

    if self.twin:
        name = "Bob"
    else:
        name = "Kevin"

    if self.who_i_play == 'X':
        if diff < 0:
            new_remark = f"{name} is proud of himself. He just gained {opp} points"
        elif diff > 0:
            new_remark = f"{name} feels stupa. He lost {diff} points"
    else:
        if diff > 0:
            new_remark = f"Papoy papoy! {name} got {diff} points from that turn!"
        elif diff < 0:
            new_remark = f"Uh oh. {name} lost {diff} points"
            
    if new_remark == '':
            new_remark = random_remark(self)

    return new_remark
        

def random_remark(self):
    random_int = random.randint(1,5)

    new_remark = ''

    if self.twin:
        name = "Bob"
    else:
        name = "Kevin"

    if random_int == 1:
        new_remark = "Ba ba ba ba banana"
    elif random_int == 2:
        new_remark = "Bello"
    elif random_int == 3:
        new_remark = "Papoy papoy"
    elif random_int == 4:
        new_remark = "Poopaye!"
    else:
        new_remark = "Miladooooooooooooo"
        
    return new_remark

def random_score_remark(self):
    random_int = random.randint(1,5)

    if self.twin:
        name = "Bob"
    else:
        name = "Kevin"

    new_remark = ''
    if random_int == 1:
        new_remark = f"{name} just called you Stupa! He's winning now"
    elif random_int == 2:
        new_remark = f"{name} says ayooooooooooooo! He's winning for this turn"
    elif random_int == 3:
        new_remark = f"{name} is winning. He's very happy"
    else: 
        new_remark = f"{name} wants you to try harder"

    return new_remark
    
# OPTIONAL THINGS TO KEEP TRACK OF:

#  WHO_MY_OPPONENT_PLAYS = other(WHO_I_PLAY)
#  MY_PAST_UTTERANCES = []
#  OPPONENT_PAST_UTTERANCES = []
#  UTTERANCE_COUNT = 0
#  REPEAT_COUNT = 0 or a table of these if you are reusing different utterances

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
            news = do_move(state, i, j, o)
            yield [(i, j), news]

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
        news = game_types.State(old=state)
        news.board[i][j] = state.whose_move
        news.whose_move = o
        return news

# The main adversarial search function:
def minimax(self,
        state,
        depth_remaining,
        pruning = False,
        alpha = None,
        beta = None,
        time_limit = None, autograding = False, special_static_eval_fn=None):
    
    start_time = time.time()
    newValMax = 0
    newValMin = 0

    if depth_remaining == 0 or done(state):
        self.num_static_evals_this_turn += 1
        if autograding:
            return [None, special_static_eval_fn(state)]
        else:
            return [None, self.static_eval(state, GAME_TYPE)]
      
    max_player = state.whose_move == "X"

    if max_player:
        maxVal = float('-inf')
        max_move = None
        

        for item in move_gen(state):
            move = item[0]
            new_state = item[1]
            
            if time_limit != None and time.time() - start_time >= time_limit:
                print("Time's up!")
                if max_move == None:
                    max_move = move
                break
            
            _, newValMax = minimax(self, new_state, depth_remaining - 1, pruning,
                                alpha, beta, time_limit, autograding, special_static_eval_fn)

            if newValMax > maxVal:
                maxVal = newValMax
                max_move = move
            
            if pruning:
                alpha = max(newValMax, alpha)
                if beta <= alpha:
                    self.alpha_beta_cutoffs_this_turn += 1
                    break
                                
        return (max_move, maxVal)
     
    else:
        provisional = float('inf')
        min_move = None

        for item in move_gen(state):
            move = item[0]
            new_state = item[1]
            
            if time_limit != None and time.time() - start_time >= time_limit:
                if min_move == None:
                    min_move = move
                break

            _, newValMin = minimax(self, new_state, depth_remaining - 1, pruning,
                                alpha, beta, time_limit, autograding, special_static_eval_fn)

            if newValMin < provisional:
                provisional = newValMin
                min_move = move
            
            if pruning:
                beta = min(newValMin, beta)
                if beta <= alpha:
                    self.alpha_beta_cutoffs_this_turn += 1
                    break
                
        print("\n", provisional)
        print(min_move, "\n")
        
        return (min_move, provisional)
        
    # Only the score is required here but other stuff can be returned
    # in the list, after the score, in case you want to pass info
    # back from recursive calls that might be used in your utterances,
    # etc. 


def check_directions(self, i, j, board, character, k):
    totalScore = 0
    forwards = 0
    down = 0
    diagonal = 0
    diagonal1 = 0
    count = j
    countBack = j
    whiteSpace = False

    numtowin = k

    col_fit = True
    row_fit = True

    back_col_fit = True
    back_row_fit = True

    if j + (numtowin - 1) >= len(board[0]):
        col_fit = False

    if i + (numtowin - 1) >= len(board):
        row_fit = False

    if j - (numtowin - 1) < 0:
        back_col_fit = False

    if i + (numtowin  - 1) >= len(board):
        back_row_fit = False
        
    if col_fit:
        for x in range (j, j + (numtowin)):
            if board[i][x] != ' ' and board[i][x] != character:
                forwards = 0
                whiteSpace = False
                break
                
            if board[i][x] == character:
                forwards += 1

                if j-1 >= 0 and j + numtowin - 1 < len(board[0]) and board[i][j - 1] == ' '  and board[i][j + numtowin - 1] == ' ':
                   whiteSpace = True           
                    
        totalScore += score(self, forwards, numtowin)
        whiteSpace = False
    
    if row_fit:
        for x in range (i, i + (numtowin)):
            if board[x][j] != ' ' and board[x][j] != character:
                down = 0
                whiteSpace = False
                break
            
            if board[x][j] == character:
                down += 1
                    
        totalScore += score(self, down, numtowin)
        whiteSpace = False

    if back_col_fit and back_row_fit:
        for x in range(i, i + (numtowin)):
            if board[x][countBack] != ' ' and board[x][countBack] != character:
                diagonal1 = 0
                whiteSpace = False
                break

            if board[x][countBack] == character:
                diagonal1 += 1
                    
            countBack -= 1

            
        totalScore += score(self, diagonal1, numtowin)
        whiteSpace = False
    
    if col_fit and row_fit:
        for x in range(i, i + (numtowin)):
            if board[x][count] != ' ' and board[x][count] != character:
                diagonal = 0
                whiteSpace = False
                break

            if board[x][count] == character:
                diagonal += 1
                
            count += 1
            
        totalScore += score(self, diagonal, numtowin)
        whiteSpace = False 
        
    return totalScore

def score(self, num_connected, k):
    multiplier = 0
    score = 0
    
    if self.twin:
        if num_connected == k:
            score = 10000
        elif num_connected == k - 1:
            score = 500
        elif num_connected == k - 2:
            score = 50
        else:
            score = num_connected * multiplier
    else:
        if num_connected == k:
            score = 10000
        elif num_connected == k - 1:
            score = 500
        elif num_connected == k - 2:
            score = 50
        else:
            score = num_connected * multiplier
        
    return score

def done(state):
    board = state.board

    finished = True
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == ' ':
                finished = False
    return finished