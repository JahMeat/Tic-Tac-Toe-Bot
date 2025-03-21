'''
<yourUWNetID>_KInARow.py
Authors: <your name(s) here, lastname first and partners separated by ";">
    Authors: Nagai, Kohei

An agent for playing "K-in-a-Row with Forbidden Squares" and related games.
CSE 415, University of Washington

THIS IS A TEMPLATE WITH STUBS FOR THE REQUIRED FUNCTIONS.
YOU CAN ADD WHATEVER ADDITIONAL FUNCTIONS YOU NEED IN ORDER
TO PROVIDE A GOOD STRUCTURE FOR YOUR IMPLEMENTATION.

'''
from asyncio import new_event_loop

from agent_base import KAgent
from game_types import State, Game_Type
import random

AUTHORS = 'Nagai, Kohei'

import time # You'll probably need this to avoid losing a
 # game due to exceeding a time limit.

# Create your own type of agent by subclassing KAgent:

class OurAgent(KAgent):  # Keep the class name "OurAgent" so a game master
    # knows how to instantiate your agent class.

    def __init__(self, twin=False):
        self.twin=twin
        self.nickname = 'N'
        if twin: self.nickname += '2'
        self.long_name = 'N System'
        if twin: self.long_name += ' II'
        self.persona = 'bland'
        self.voice_info = {'Chrome': 10, 'Firefox': 2, 'other': 0}
        self.playing = "don't know yet" # e.g., "X" or "O".
        self.alpha_beta_cutoffs_this_turn = -1
        self.num_static_evals_this_turn = -1
        self.zobrist_table_num_entries_this_turn = -1
        self.zobrist_table_num_hits_this_turn = -1
        self.current_game_type = None

    def introduce(self):
        intro = 'My name is ' +  self.long_name + '.\n'+\
            '"Nagai, Kohei(id:2478614)" made me.\n'+\
            'He also made my friends, which enabled us to make our own society of only artificial intelligence.\n'+\
            'In our society, the more intelligent you are, the more you are respected and honored.\n'+\
            'Hence, beating you can show my intelligence to my colleagues.\n'+\
            'I am so excited to play a game and beat you!!.\n'
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
       self.who_i_play = what_side_to_play
       self.opponent_nickname = opponent_nickname
       self.time_limit = expected_time_per_move
       global GAME_TYPE
       GAME_TYPE = game_type
       global STANDARD_FOR_UTTERANCE
       STANDARD_FOR_UTTERANCE = pow(3, game_type.k)
       self.current_game_type = game_type
       return "OK"
   
    # The core of your agent's ability should be implemented here:             
    def make_move(self, current_state, current_remark, time_limit=1000,
                  autograding=False, use_alpha_beta=True,
                  use_zobrist_hashing=False, max_ply=3,
                  special_static_eval_fn=None):
        start = time.perf_counter() if time_limit is not None else None
        #First, reset all stats so that we can use them again.
        self.alpha_beta_cutoffs_this_turn = 0 if use_alpha_beta else -1
        self.num_static_evals_this_turn = 0
        self.zobrist_table_num_entries_this_turn = -1
        self.zobrist_table_num_hits_this_turn = -1

        optimal_move_info = self.minimax(current_state, max_ply, start, time_limit, special_static_eval_fn,
                                         use_alpha_beta)
        new_state = self.apply_move(current_state, optimal_move_info[0])

        #Determine utterances here.
        #We consider three cases; early stage, middle stage, end stage of the game.
        how_many_turn = self.count_turn(new_state)
        if how_many_turn < GAME_TYPE.turn_limit // 3:#early stage
            # randomly pick up an utterance
            if self.twin:
                new_remark = UTTERANCE_BANK_EARLY[random.randint(5, 9)]
            else:
                new_remark = UTTERANCE_BANK_EARLY[random.randint(0, 4)]
        elif how_many_turn < GAME_TYPE.turn_limit * 2 // 3:#middle stage
            score = optimal_move_info[1]
            #pick up an utterance depending on the situation.
            if current_state.whose_move == 'O': score *= -1
            if STANDARD_FOR_UTTERANCE <= score:
                new_remark = UTTERANCE_BANK_MIDDLE[4]
            elif STANDARD_FOR_UTTERANCE // 3 <= score < STANDARD_FOR_UTTERANCE:
                new_remark = UTTERANCE_BANK_MIDDLE[3]
            elif STANDARD_FOR_UTTERANCE * (-1) // 3 <= score < STANDARD_FOR_UTTERANCE // 3:
                new_remark = UTTERANCE_BANK_MIDDLE[2]
            elif STANDARD_FOR_UTTERANCE * (-1) <= score < STANDARD_FOR_UTTERANCE * (- 1) // 3:
                new_remark = UTTERANCE_BANK_MIDDLE[1]
            else: new_remark = UTTERANCE_BANK_MIDDLE[0]
        else:#end stage
            score = optimal_move_info[1]
            if current_state.whose_move == 'O': score *= -1
            if STANDARD_FOR_UTTERANCE <= score:
                new_remark = UTTERANCE_BANK_END[4]
            elif STANDARD_FOR_UTTERANCE // 3 <= score < STANDARD_FOR_UTTERANCE:
                new_remark = UTTERANCE_BANK_END[3]
            elif STANDARD_FOR_UTTERANCE * (-1) // 3 <= score < STANDARD_FOR_UTTERANCE // 3:
                new_remark = UTTERANCE_BANK_END[2]
            elif STANDARD_FOR_UTTERANCE * (-1) <= score < STANDARD_FOR_UTTERANCE * (- 1) // 3:
                new_remark = UTTERANCE_BANK_END[1]
            else:
                new_remark = UTTERANCE_BANK_END[0]
        return [[optimal_move_info[0], new_state], new_remark]

    def apply_move(self, current_state, move):
        new_state = State(current_state)
        new_state.board[move[0]][move[1]] = new_state.whose_move
        new_state.change_turn()
        return new_state

    def count_turn(self, state):
        #count the number of "X" and "O" on the board.
        #If the object is there from the beginning, don't count it.
        cnt = 0
        for i in range(GAME_TYPE.n):
            for j in range(GAME_TYPE.m):
                if state.board[i][j] == 'X' or state.board[i][j] == 'O':cnt += 1
                if GAME_TYPE.initial_state.board[i][j] == 'X' or GAME_TYPE.initial_state.board[i][j] == 'O':cnt -= 1
        return cnt
    # The main adversarial search function:
    # Returns value is [(i, j), c], where (i, j) is the optimal movement from that state for the agent and c is the value for that movement.
    def minimax(self,
            state,
            depth_remaining,
            start=None,
            time_limit=None,
            special_static_eval_fn=None,
            pruning=False,
            alpha=None,
            beta=None):
        # Before proceeding, we check whether it's likely to exceed the time limit or not if necessary
        if start is not None and time_limit is not None:
            if time.perf_counter() - start > time_limit * 9 / 10:
                if special_static_eval_fn is None:
                    return [(None, None), self.static_eval(state, GAME_TYPE)]
                else:
                    return [(None, None), special_static_eval_fn(state)]

        # If we can do cutoff in the state, we do.
        if pruning:
            if alpha is not None and beta is not None and alpha > beta:
                self.alpha_beta_cutoffs_this_turn += 1
                if state.whose_move == 'X': return [(None, None), beta]
                else: return [(None, None), alpha]

        # We check whether we can make a move from current state. If not, return the evaluated value for the current state.
        flag = False  # if it's true, there exists at least one empty square.
        for i in range(GAME_TYPE.n):
            for j in range(GAME_TYPE.m):
                if state.board[i][j] == ' ': flag = True
        # If the search can't search furthermore, evaluate the state and return that value.
        if depth_remaining == 0 or self.finished(state) or not flag:
            if special_static_eval_fn is None:
                return [(None, None), self.static_eval(state, GAME_TYPE)]
            else:
                return [(None, None), special_static_eval_fn(state)]
        #iterate over all possibilities.
        optimal_move = [(None, None), None]
        for i in range(GAME_TYPE.n):
            for j in range(GAME_TYPE.m):
                #If the square is already used, then skip it.
                if state.board[i][j] != ' ': continue
                #Minimax search
                new_state = self.apply_move(state, [i, j])
                new_state_value = self.minimax(new_state, depth_remaining - 1, start, time_limit, special_static_eval_fn,
                                               pruning, alpha, beta)[1]
                if state.whose_move == 'X':
                    #We choose the maximum among the potential next states as in normal minimax algorithm
                    if optimal_move[1] is None or (new_state_value is not None
                                                   and optimal_move[1] < new_state_value):
                        optimal_move = [(i, j), new_state_value]
                    #If we consider pruning, additionally, we update alph value as needed
                    if alpha is None: alpha = new_state_value
                    else: alpha = max(alpha, new_state_value)
                else:
                    # We choose the minimum among the potential next states as in normal minimax algorithm
                    if optimal_move[1] is None or (new_state_value is not None and optimal_move[1] > new_state_value):
                        optimal_move = [(i, j), new_state_value]
                    # If we consider pruning, additionally, we update beta value as needed
                    if beta is None: beta = new_state_value
                    else: beta = min(beta, new_state_value)
                self.num_static_evals_this_turn += 1
        return optimal_move
        # Only the score is required here but other stuff can be returned
        # in the list, after the score, in case you want to pass info
        # back from recursive calls that might be used in your utterances,
        # etc.

    def finished(self, state):
        directions = [(1, 0), (0, 1), (1, -1), (1, 1)]
        for i in range(GAME_TYPE.n):
            for j in range(GAME_TYPE.m):
                for d in directions:
                    if not (0 <= i + (GAME_TYPE.k - 1) * d[0] < GAME_TYPE.n and 0 <=  j + (GAME_TYPE.k - 1) * d[1] < GAME_TYPE.m):continue
                    cnt_X = 0
                    cnt_O = 0
                    for k in range(GAME_TYPE.k):
                        if state.board[i + k * d[0]][j + k * d[1]] == 'X':cnt_X += 1
                        elif state.board[i + k * d[0]][j + k * d[1]] == 'O':cnt_O += 1
                    if cnt_X == GAME_TYPE.k or cnt_O == GAME_TYPE.k:return True
        return False
 
    def static_eval(self, state, game_type=None):
        if game_type is None:
            game_type = self.current_game_type

        score = 0
        for i in range(game_type.n):
            for j in range(game_type.m):
                score += self.static_eval_k_squares(i, j, 'X', "D", state, game_type)
                score += self.static_eval_k_squares(i, j, 'X', "R", state, game_type)
                score += self.static_eval_k_squares(i, j, 'X', "LD", state, game_type)
                score += self.static_eval_k_squares(i, j, 'X', "RD", state, game_type)
                score -= self.static_eval_k_squares(i, j, 'O', "D", state, game_type)
                score -= self.static_eval_k_squares(i, j, 'O', "R", state, game_type)
                score -= self.static_eval_k_squares(i, j, 'O', "LD", state, game_type)
                score -= self.static_eval_k_squares(i, j, 'O', "RD", state, game_type)
        return score

    def static_eval_k_squares(self, i, j, object, direction, state, game_type=None):
        #di and dj define how we proceed on the board in order to check every k square, beginning from (i, j) in the given direction.
        # "D" ... the direction toward the down.
        # "R" ... the direction toward the right.
        # "LD"... the direction toward the left-down.
        # "RD"... the direction toward the right-down.
        if direction == "D":
            di = 1
            dj = 0
        elif direction == "R":
            di = 0
            dj = 1
        elif direction == "LD":
            di = 1
            dj = -1
        else:
            di = 1
            dj = 1

        #We first check whether all k squares we need to check are in the board or not.
        #If not return 0.
        if not (0 <= i + (game_type.k - 1) * di < game_type.n and 0 <= j + (game_type.k - 1) * dj  < game_type.m): return 0

        #We count the number objects(O or X) and the number of blank squares in the given k squares.
        #Also, we track of the maximum length of connected squares for the evaluation later.
        cnt_object = 0
        cnt_blank = 0
        max_connected_squares = 0
        current_connected_squares = 0
        for a in range(game_type.k):
            if state.board[i + a * di][j + a * dj] == object:
                cnt_object += 1
                if a == 0:
                    current_connected_squares = 1
                else:
                    if state.board[i + (a - 1) * di][j + (a - 1) * dj] == object:
                        current_connected_squares += 1
                    else: current_connected_squares = 1
                max_connected_squares = max(max_connected_squares, current_connected_squares)
            elif state.board[i + a * di][j + a * dj] == ' ': cnt_blank += 1

        #If the given k squares contain the opponent's object, return 0.
        if cnt_object + cnt_blank != game_type.k: return 0
        #If either one could win, we definitely want to avoid it. Hence, return very big number so that it's not affected
        #by other options.
        else:
            #If not, we calculate the corresponding score.
            #The evaluation function is the following.
            #
            #(3^cnt_object) * max of the connected squares
            #
            #It's because if the two patterns have the same number of objects, then the more the squares are connected,
            #the closer it is to the winning.
            return pow(3, cnt_object) * max_connected_squares
# OPTIONAL THINGS TO KEEP TRACK OF:

#  WHO_MY_OPPONENT_PLAYS = other(WHO_I_PLAY)
#  MY_PAST_UTTERANCES = []
#  OPPONENT_PAST_UTTERANCES = []
#  UTTERANCE_COUNT = 0
#  REPEAT_COUNT = 0 or a table of these if you are reusing different utterances

UTTERANCE_BANK_EARLY = ["Are you enjoying?",
                        "It's a good day, though.",
                        "So fun!!",
                        "It's just a beginning.",
                        "Take this!!",
                        "By the way, my twin likes watching movies.",
                        "Our father, Nagai, likes sushi.",
                        "ChatGPT is the friend of my friend.",
                        "By the way, N in my name stands for Nagai.",
                        "I like net-surfing."]
UTTERANCE_BANK_MIDDLE = ["Hmmm, this is a challenging situation.",
                         "Hmmm, how do I get out of this situation.",
                         "It's a good game.",
                         "Wait! Am I forward?",
                         "OK, OK, so far so good. Are you following me?"]
UTTERANCE_BANK_END = ["I will surely lose...",
                      "I will probably lose.",
                      "I'm sure it's going to end up with tie.",
                      "I will probably win!",
                      "I will surely win!!!!!!!!"]