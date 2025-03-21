import time 
import random
from agent_base import KAgent
from game_types import State, Game_Type

class OurAgent(KAgent): 

    def __init__(self, twin=False):
        self.twin = twin
        self.nickname = 'Jah' if not twin else 'Goofy'
        self.long_name = 'Notorious J.A.H' if not twin else 'This Goofy Clown'
        self.persona = 'rapper'
        self.voice_info = {'Chrome': 10, 'Firefox': 2, 'other': 0}
        self.playing = "don't know yet"
        self.alpha_beta_cutoffs_this_turn = -1
        self.num_static_evals_this_turn = -1
        self.zobrist_table_num_entries_this_turn = -1
        self.zobrist_table_num_hits_this_turn = -1
        self.current_game_type = None
        self.zobrist = None  # Initialize Zobrist instance

    def introduce(self):
        intro = '\nWAZZUP MI GENTES. This yo boy Notorious J.A.H speaking.\n'+\
            'Imma show you how I roll with a bombastic game of tic-tac-toe.\n'
        if self.twin: intro = "I'm the goofy clown that this glorious hoodlum is playing against.\n"
        return intro

    def prepare(self, game_type, what_side_to_play, opponent_nickname, expected_time_per_move=0.1, utterances_matter=True):
        self.current_game_type = game_type
        self.playing = what_side_to_play
        self.opponent = opponent_nickname
        self.time = expected_time_per_move
        self.zobrist = Zobrist(game_type.n, game_type.m)
        return "OK"

    def generate_utterance(self, move, score, state):
        rap_responses = [
            f"Yo {self.opponent}, that move was weak! Step up, or step out. I got the crown, no doubt. 🎤",
            f"Ha! That move was softer than a marshmallow. You best bring heat if you wanna beat the {self.nickname}.",
            f"Ooooh, I saw that! Thought you was slick? Nah, son, you just fell into my trap.",
            f"That move was rookie-level, straight outta a beginner’s playbook. Try again, {self.opponent}!",
            f"You tryna cross me? Nah, I’m the King of this board. Bow down to the champion!",
            f"Check this out—every move I make is a headline, every step you take is a flatline.",
            f"Oof, that was a bad move! Should I give you a lil’ tutorial, or are we good? 😏",
            f"I ain't playin' chess, but I just checked you, mate. Watch and learn!",
            f"You think you’re on my level? Psh, I’ve already mapped out ten moves ahead.",
            f"That was your big plan? Man, my grandma got better strats than that!",
            f"I got 99 problems, but this game ain't one. Better come correct next turn!",
            f"Yo, {self.opponent}, you might as well surrender now. I’m just warmin’ up!",
            f"Ayo, that move was cute. You tryna make this a freestyle battle or what?",
            f"Biggie vibes—every move I make is legendary, and every move you make is... well, let’s just say tragic. 😆",
            f"Call the ref, ‘cause that was a foul move! Ain’t no way you’re takin’ this W from me.",
            f"Yo, {self.opponent}, you shoulda stayed in school ‘cause these Tic-Tac-Toe lessons ain't free!",
            f"Check the scoreboard, check the rhyme, check yourself—'cause this game’s already mine!",
            f"Ayy {self.opponent}, you tryna battle me on this board? You better come correct, ‘cause I play like a legend.",
            f"Man, I’m runnin’ this board like it’s my kingdom. You just a visitor, {self.opponent}.",
            f"That move was sloppy—lookin’ like a rookie. I’m over here droppin’ classics, and you out here doodlin’.",
            f"Yo {self.opponent}, that move was more predictable than a nursery rhyme. Step it up!",
            f"Ha! You thought that move was fire? More like a candle. I’m bringin’ the whole inferno. 🔥",
            f"You see that move? That’s strategy. That’s skill. That’s why I run this board.",
            f"You playin’ like you just installed the game, {self.opponent}. You sure you wanna keep goin’?",
            f"This ain’t just Tic-Tac-Toe, this a chessboard for lyrical geniuses. You ain’t ready for this.",
            f"Not gonna lie, I respect the hustle, {self.opponent}. But respect ain’t gonna win you this game. I am.",
            f"You ever seen an artist paint a masterpiece? That’s what I’m doin’ here, and you just tryna color inside the lines.",
            f"I ain't even sweatin’, {self.opponent}. You better bring a bigger playbook, ‘cause I seen this all before.",
            f"Yo, this a freestyle battle or a game? ‘Cause the way I’m winning, you gon’ need a mic instead of a marker.",
            f"You movin’ slow like dial-up internet. Meanwhile, I’m makin’ moves at lightning speed. ⚡",
            f"I coulda won this game blindfolded, but I wanted you to see me takin’ this W in 4K.",
            f"You just played that move? Bruh. I was worried for a sec, but now I know this game’s already mine.",
        ]

        # Adjust response based on score
        if score > 500:
            rap_responses.append(f"Yo {self.opponent}, I ain't seen a challenge yet. Might as well call me undefeated.")
        elif score < -500:
            rap_responses.append(f"Wait—am I losin’? Nah, nah, that’s fake news. I’m about to drop a comeback track!")
        elif score == 0:
            rap_responses.append(f"We neck and neck? Nah, gimme one more move, and it’s game over for you.")

        # Adjust response based on game state
        if state.finished:
            if score > 900:
                return f"BOOM! That’s game, {self.opponent}. That L you took? That’s a platinum hit!"
            elif score < -900:
                return f"Wait… how did I lose? Somebody check the script—I don’t take L’s!"
            else:
                return f"It’s a draw? Nah, we runnin’ it back. I don’t do ties, only victories!"

        # Randomly select a response
        return random.choice(rap_responses)

    def make_move(self, current_state, current_remark, time_limit=1000, 
                  autograding=False, use_alpha_beta=True, 
                  use_zobrist_hashing=True, max_ply=3, 
                  special_static_eval_fn=None):
        self.num_static_evals_this_turn = 0
        self.alpha_beta_cutoffs_this_turn = 0
        self.zobrist.cache = {}

        start_time = time.time()
        best_score, best_move = self.minimax(current_state, max_ply, pruning=use_alpha_beta,
                                             special_static_eval_fn=special_static_eval_fn,
                                             start_time=start_time, time_limit=time_limit, 
                                             use_zobrist_hashing=use_zobrist_hashing)
        
        if best_move is None:
            legal_moves = self.generate_moves(current_state)
            best_move = legal_moves[0] if legal_moves else None

        new_state = State(old=current_state)
        if best_move:
            marker = current_state.whose_move
            new_state.board[best_move[0]][best_move[1]] = marker
            new_state.change_turn()

        new_remark = self.generate_utterance(best_move, best_score, current_state)
        return [[best_move, new_state], new_remark]

    def minimax(self, 
                state, 
                depth_remaining, 
                pruning=False, 
                alpha=None, 
                beta=None, 
                special_static_eval_fn=None, 
                start_time=None, 
                time_limit=None,
                use_zobrist_hashing=False):
        
        if use_zobrist_hashing:
            cached_score = self.zobrist.lookup(state.board)
            if cached_score is not None:
                return [cached_score, None]
    
        if time_limit is not None and time.time() - start_time > time_limit:
            eval_fn = special_static_eval_fn if special_static_eval_fn else lambda s: self.static_eval(s, self.current_game_type)
            self.num_static_evals_this_turn += 1
            return [eval_fn(state), None]

        if depth_remaining == 0 or state.finished:
            eval_fn = special_static_eval_fn if special_static_eval_fn else lambda s: self.static_eval(s, self.current_game_type)
            self.num_static_evals_this_turn += 1
            return [eval_fn(state), None]

        if pruning:
            if alpha is None:
                alpha = -float('inf')
            if beta is None:
                beta = float('inf')

        legal_moves = self.generate_moves(state)
        if not legal_moves:
            eval_fn = special_static_eval_fn if special_static_eval_fn else lambda s: self.static_eval(s, self.current_game_type)
            self.num_static_evals_this_turn += 1
            return [eval_fn(state), None]
        
        best_move = None
        best_score = -float('inf') if state.whose_move == "X" else float('inf')
        
        for move in legal_moves:
            if time_limit is not None and time.time() - start_time > time_limit:
                break  
            
            new_state = State(old=state)
            new_state.board[move[0]][move[1]] = state.whose_move
            new_state.change_turn()
            
            score = self.minimax(new_state, depth_remaining - 1, pruning, alpha, beta,
                                 special_static_eval_fn, start_time, time_limit)[0]
            
            if state.whose_move == "X":
                if score > best_score:
                    best_score = score
                    best_move = move
                if pruning:
                    alpha = max(alpha, best_score)
                    if beta <= alpha:
                        self.alpha_beta_cutoffs_this_turn += 1
                        break
            else:
                if score < best_score:
                    best_score = score
                    best_move = move
                if pruning:
                    beta = min(beta, best_score)
                    if beta <= alpha:
                        self.alpha_beta_cutoffs_this_turn += 1
                        break

        self.zobrist.store(state.board, best_score)
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

class Zobrist:
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.table = {((r, c, p)): random.getrandbits(64) for r in range(n) for c in range(m) for p in ["X", "O"]}
        self.cache = {}

    def compute_hash(self, board):
        h = 0
        for r in range(self.n):
            for c in range(self.m):
                piece = board[r][c]
                if piece in ["X", "O"]:
                    h ^= self.table[(r, c, piece)]
        return h

    def store(self, board, score):
        h = self.compute_hash(board)
        self.cache[h] = score

    def lookup(self, board):
        h = self.compute_hash(board)
        return self.cache.get(h, None)

