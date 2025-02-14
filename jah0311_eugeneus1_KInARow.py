'''
<yourUWNetID>_KInARow.py
Authors: <your name(s) here, lastname first and partners separated by ";">
  Example:  
    Authors: Chen, Jah; Cheung, Eugene

An agent for playing "K-in-a-Row with Forbidden Squares" and related games.
CSE 415, University of Washington

THIS IS A TEMPLATE WITH STUBS FOR THE REQUIRED FUNCTIONS.
YOU CAN ADD WHATEVER ADDITIONAL FUNCTIONS YOU NEED IN ORDER
TO PROVIDE A GOOD STRUCTURE FOR YOUR IMPLEMENTATION.

'''

from agent_base import KAgent
from game_types import State, Game_Type

AUTHORS = 'Jane Smith and Laura Lee' 

import time # You'll probably need this to avoid losing a
 # game due to exceeding a time limit.

# Create your own type of agent by subclassing KAgent:

class OurAgent(KAgent):  # Keep the class name "OurAgent" so a game master
    # knows how to instantiate your agent class.

    def __init__(self, twin=False):
        self.twin=twin
        self.nickname = 'Jah'
        if twin: self.nickname = 'Eugene'
        self.long_name = 'Jah Chen'
        if twin: self.long_name = 'Eugene Cheung'
        self.persona = 'bland'
        self.voice_info = {'Chrome': 10, 'Firefox': 2, 'other': 0}
        self.playing = "don't know yet" # e.g., "X" or "O".
        self.alpha_beta_cutoffs_this_turn = -1
        self.num_static_evals_this_turn = -1
        self.zobrist_table_num_entries_this_turn = -1
        self.zobrist_table_num_hits_this_turn = -1
        self.current_game_type = None

    def introduce(self):
        intro = '\nMy name is Jahgene!.\n'+\
            'My intelligence capacity is controlled by two personalities: Jah and Eugene.\n'+\
            'The NetID for Jah is "jah0311" and for Eugene is "eugeneus1".\n'
        if self.twin: intro += "By the way, I'm the TWIN.\n"
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
       self.current_game_type = game_type  # Set the current game type for later use.
       return "OK"
           
    def make_move(self, current_state, current_remark, time_limit=1000,
                  autograding=False, use_alpha_beta=True,
                  use_zobrist_hashing=False, max_ply=3,
                  special_static_eval_fn=None):
    
        best_score, best_move = self.minimax(current_state, max_ply, pruning=use_alpha_beta, special_static_eval_fn=special_static_eval_fn)
        if best_move is None:
            legal_moves = self.generate_moves(current_state)
            best_move = legal_moves[0] if legal_moves else None

        new_state = State(old=current_state)
        if best_move:
            marker = current_state.whose_move
            new_state.board[best_move[0]][best_move[1]] = marker
            new_state.change_turn()

        new_remark = f"I chose move {best_move} with score {best_score}."
       
        return [[best_move, new_state], new_remark]

    def minimax(self,
                state,
                depth_remaining,
                pruning=False,
                alpha=None,
                beta=None,
                special_static_eval_fn=None):
    
        if depth_remaining == 0 or state.finished:
            eval_fn = special_static_eval_fn if special_static_eval_fn is not None else (lambda s: self.static_eval(s, self.current_game_type))
            return [eval_fn(state), None]

        if pruning:
            if alpha is None:
                alpha = -float('inf')
            if beta is None:
                beta = float('inf')
        else:
            alpha = None
            beta = None
                
        legal_moves = self.generate_moves(state)
        if not legal_moves:
            eval_fn = special_static_eval_fn if special_static_eval_fn is not None else (lambda s: self.static_eval(s, self.current_game_type))
            return [eval_fn(state), None]
        
        best_move = None

        if state.whose_move == "X":
            best_score = -float('inf')
            for move in legal_moves:
                new_state = State(old=state)
                new_state.board[move[0]][move[1]] = "X"
                new_state.change_turn()
                score = self.minimax(new_state, depth_remaining - 1, pruning, alpha, beta, special_static_eval_fn=special_static_eval_fn)[0]
                if score > best_score:
                    best_score = score
                    best_move = move
                if pruning:
                    alpha = max(alpha, best_score)
                    if beta <= alpha:
                        break
            return [best_score, best_move]
        else: 
            best_score = float('inf')
            for move in legal_moves:
                new_state = State(old=state)
                new_state.board[move[0]][move[1]] = "O"
                new_state.change_turn()
                score = self.minimax(new_state, depth_remaining - 1, pruning, alpha, beta, special_static_eval_fn=special_static_eval_fn)[0]
                if score < best_score:
                    best_score = score
                    best_move = move
                if pruning:
                    beta = min(beta, best_score)
                    if beta <= alpha:
                        break
            return [best_score, best_move]

    def generate_moves(self, state):
        board = state.board
        moves = []
        nrows = len(board)
        ncols = len(board[0])
        for i in range(nrows):
            for j in range(ncols):
                if board[i][j] == " ":
                    moves.append((i, j))
        return moves

    def static_eval(self, state, game_type=None):
        board = state.board
        win_length = game_type.k
        nrows = len(board)
        ncols = len(board[0])
        score = 0

        for i in range(nrows):
            score += self.sequence_eval(board[i], win_length)

        for j in range(ncols):
            score += self.sequence_eval([board[i][j] for i in range(nrows)], win_length)

        for i in range(nrows):
            for j in range(ncols):
                diag = []
                k = 0
                while (i+k) < nrows and (j+k) < ncols:
                    diag.append(board[i+k][j+k])
                    k += 1
                if len(diag) >= win_length:
                    score += self.sequence_eval(diag, win_length)

        for i in range(nrows):
            for j in range(ncols):
                diag = []
                k = 0
                while (i+k) < nrows and (j-k) >= 0:
                    diag.append(board[i+k][j-k])
                    k += 1
                if len(diag) >= win_length:
                    score += self.sequence_eval(diag, win_length)

        return score

    def sequence_eval(self, sequence, win_length):
        score = 0
        n = len(sequence)
        
        for i in range(n - win_length + 1):
            window = sequence[i:i+win_length]
            if "-" in window:  
                continue
            x_count = window.count("X")
            o_count = window.count("O")
            
            if x_count == win_length:
                return 10 ** win_length
            elif o_count == win_length:
                return -10 ** win_length
            
            if x_count > 0 and o_count == 0:
                score += 10 ** (x_count - 1)
            elif o_count > 0 and x_count == 0:
                score -= 10 ** (o_count - 1)
        return score

 
# OPTIONAL THINGS TO KEEP TRACK OF:

#  WHO_MY_OPPONENT_PLAYS = other(WHO_I_PLAY)
#  MY_PAST_UTTERANCES = []
#  OPPONENT_PAST_UTTERANCES = []
#  UTTERANCE_COUNT = 0
#  REPEAT_COUNT = 0 or a table of these if you are reusing different utterances

