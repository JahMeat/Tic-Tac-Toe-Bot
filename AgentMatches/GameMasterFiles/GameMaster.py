'''GameMaster.py
 based on code from RunKInARow.py

 Updated Nov. 17, 2024, to use asyncio.  (Required when
on the web, to allow screen to update after each move.)

For further info on this, see:
https://jeff.glass/post/pyscript-realtime-page-updates/

'''

from pyodide.ffi import create_proxy, to_js
from asyncio import sleep, ensure_future
import js
TIME_PER_MOVE = 1.0 # Constant for now.

USE_HTML = False
INITIAL_STATE = None
GM_VOICE = None
PX_VOICE = None
PO_VOICE = None

UTTERANCE_COUNT = None # used in asyncTalk functionality.

from winTesterForK import winTesterForK

# Establish global variables, with defaults for now.
K = 3
N = 3
M = 3
GAME_TYPE = None
TURN_LIMIT = 9
USE_VOICES = False # updated by WebGameMaster.py

# To be called from WebGameAgent if using the web:
def set_game(game_type):
    global K, GAME_TYPE, TURN_LIMIT, N, M, INITIAL_STATE
    K = game_type.k
    N = game_type.n
    M = game_type.m
    GAME_TYPE = game_type
    TURN_LIMIT = game_type.turn_limit
    INITIAL_STATE = game_type.initial_state
    
PLAYERX = None
PLAYERO = None
def set_players(px, po):
    global PLAYERX, PLAYERO
    PLAYERX = px
    PLAYERO = po
    if USE_VOICES: prep_voices()

#FIRST_UTTERANCE = True
async def utter(role, text):
    global GM_VOICE, PX_VOICE, PO_VOICE
    if not USE_VOICES: return
    #global FIRST_UTTERANCE
    #if FIRST_UTTERANCE:
    #    js.alert("First utterance")
    #    FIRST_UTTERANCE = False
    if role=="GM": voice=GM_VOICE
    if role=="PX": voice=PX_VOICE
    if role=="PO": voice=PO_VOICE
#    js.talk(voice, text) # call method in voices-for-agents.js.
    await talkAndWait(voice, text)

async def talkAndWait(voice, text):
    global UTTERANCE_COUNT
    count_now = UTTERANCE_COUNT
    js.asyncTalk(voice, text, create_proxy(upCount)) # call method in voices-for-agents.js.
    i_monitoring = 0
    while UTTERANCE_COUNT == count_now:
        await sleep(0.01)
        if i_monitoring == 100:
            i_monitoring = 0
            js.console.log("waiting in talkAndWait.")
        else: i_monitoring += 1
    
def upCount():
    # Callback for asyncTalk that will increment a count when
    # a speech request finishes.
    global UTTERANCE_COUNT
    UTTERANCE_COUNT += 1
    
#utter("GM", "This is a test.")

FINISHED = False
async def runGame():
    import js
    global UTTERANCE_COUNT
    UTTERANCE_COUNT = 0
    currentState = INITIAL_STATE
    player1 = PLAYERX
    player2 = PLAYERO
    renderCommentary('The Gamemaster says: Players, introduce yourselves.')
    await utter('GM', 'Players, introduce yourselves.')
    await utter('GM', player1.nickname + ' who is playing X, says ')
    p1_intro = player1.introduce()
    renderCommentary(player1.nickname +' says: '+'  (Playing X:) '+p1_intro)
    await utter('PX', p1_intro)
    await utter('GM', player2.nickname + ' who is playing O, says ')
    p2_intro = player2.introduce()
    await utter('PO', p2_intro)
    renderCommentary(player2.nickname +' says: '+'  (Playing O:) '+player2.introduce())

#    if USE_HTML:
#        gameToHTML.startHTML(player1.nickname(), player2.nickname(), GAME_TYPE, 1)
    try:
        p1comment = player1.prepare(GAME_TYPE, 'X', player2.nickname)
    except Exception as e:
        print("Failed to prepare perhaps because: ", e)
        report = 'Player 1 ('+player1.nickname+' failed to prepare, and loses by default.'
        renderCommentary(report)
        await utter('GM', report)
        if USE_HTML: gameToHTML.reportResult(report)
        report = 'Congratulations to Player 2 ('+player2.nickname+')!'
        renderCommentary(report)
        await utter('GM', report)
        if USE_HTML: gameToHTML.reportResult(report)
        if USE_HTML: gameToHTML.endHTML()
        return
    try:
        p2comment = player2.prepare(GAME_TYPE, 'O', player1.nickname)
    except Exception as e:
        print("Failed to prepare perhaps because: ", e)
        report = 'Player 2 ('+player2.nickname+' failed to prepare, and loses by default.'
        renderCommentary(report)
        await utter('GM', report)
        if USE_HTML: gameToHTML.reportResult(report)
        report = 'Congratulations to Player 1 ('+player1.nickname+')!'
        renderCommentary(report)
        await utter('GM', report)
        if USE_HTML: gameToHTML.reportResult(report)
        if USE_HTML: gameToHTML.endHTML()
        return
        return
    report = 'We\'re playing '+GAME_TYPE.long_name+'.'
    await utter('GM', report)
    report = 'The Gamemaster says: '+report
    renderCommentary(report)
    renderCommentary('The Gamemaster says: Let\'s Play!')
    await utter('GM', "Let's play!")
    renderCommentary('The initial state is...')

    currentRemark = "The game is starting."
    await utter('GM', currentRemark)
    if USE_HTML: gameToHTML.stateToHTML(currentState)
    XsTurn = True
    name = None
    global FINISHED
    FINISHED = False
    turnCount = 0
    printState(currentState)
    js.console.log("About to start the game") # for debugging.
    while not FINISHED:
        js.console.log("Turncount = ",turnCount) # for debugging.
        who = currentState.whose_move
        if XsTurn:
            js.console.log("It's X's turn")
            playerResult = player1.make_move(currentState, currentRemark, time_limit=TIME_PER_MOVE)
            js.console.log("We are back from X")
            name = player1.nickname
            XsTurn = False
        else:
            js.console.log("It's O's turn")
            playerResult = player2.make_move(currentState, currentRemark, time_limit=TIME_PER_MOVE)
            js.console.log("We are back from O")
            name = player2.nickname
            XsTurn = True
        moveAndState, currentRemark = playerResult
        js.console.log("Back from makeMove.")
        if moveAndState==None:
            FINISHED = True; continue
        move, currentState = moveAndState
        moveReport = "Move is by "+who+" to "+str(move)
        renderCommentary(moveReport)
        utteranceReport = name +' says: '+currentRemark
        renderCommentary(utteranceReport)
        if XsTurn:
            await utter('PX', currentRemark)
        else:
            await utter('PO', currentRemark)
        if USE_HTML: gameToHTML.reportResult(moveReport)
        if USE_HTML: gameToHTML.reportResult(utteranceReport)
        possibleWin = winTesterForK(currentState, move, K)
        if possibleWin != "No win":
            FINISHED = True
            currentState.finished = True
            printState(currentState)
            if USE_HTML: gameToHTML.stateToHTML(currentState, finished=True)
            renderCommentary(possibleWin)
            await utter('GM', possibleWin)
            if USE_HTML: gameToHTML.reportResult(possibleWin)
            if USE_HTML: gameToHTML.endHTML()
            return
        printState(currentState)
        if USE_HTML: gameToHTML.stateToHTML(currentState)
        turnCount += 1
        if turnCount == TURN_LIMIT: FINISHED=True
        else:
            await sleep(WAIT_TIME_AFTER_MOVES) # NOT TOO FAST.
    printState(currentState)
    if USE_HTML: gameToHTML.stateToHTML(currentState)
    who = currentState.whose_move
    report = "Game over; it's a draw."
    renderCommentary(report)
    await utter('GM', report)
    if USE_HTML: gameToHTML.reportResult("Game Over; it's a draw")
    if USE_HTML: gameToHTML.endHTML()

def printState(s):
    global FINISHED
    board = s.board
    who = s.whose_move
    horizontalBorder = "+"+3*M*"-"+"+"
    renderCommentary(horizontalBorder)
    for row in board:
        line = "|"
        for item in row:
            line += " "+item+" "
        line += "|"
        renderCommentary(line)
    renderCommentary(horizontalBorder)
    if not FINISHED:
      renderCommentary("It is "+who+"'s turn to move.\n")

# Temporary function.  Remove when other channels are working.
def renderCommentary(stuff):
   add_to_history(stuff)
      
def add_to_history(stuff):
   # NOTE: THIS DEFN WILL BE OVERWRITTEN WHEN USED ON THE WEB.
   pass

def render_move_and_state(move, state):
   # NOTE: THIS DEFN WILL BE OVERWRITTEN WHEN USED ON THE WEB.
   print(move, state)

def render_utterance(who, utterance):
   # NOTE: THIS DEFN WILL BE OVERWRITTEN WHEN USED ON THE WEB.
   print(who+' says: '+utterance)

def get_preferred_voice(player):
    browser = js.detectBrowser()
    if hasattr(player, 'voice_prefs') and browser in player.voice_prefs:
        print(player.nickname + "'s voice_prefs: ")
        print(player.voice_prefs)
        vdesc = player.voice_prefs[browser]
        js.alert("In "+browser+" player "+player.nickname+" seems to want this voice: "+vdesc)
        try: 
            voice = js.find_voice(vdesc)
            return voice
        except Exception as e:
            js.alert("Exception when voice '"+vdesc+" in browser "+browser+" requested; Exception is in "+str(e));
            raise
            return 'DEFAULT'

    else:
        js.alert(player.nickname + " doesn't seem to have a preferred voice for browser: "+browser)
        return 'DEFAULT'

def get_browser_default_voice():
    return js.getDefaultVoice()

def get_default_PX_VOICE(playerX):
    return js.getPlayerXVoice()

def get_default_PO_VOICE(playerY):
    return js.getPlayerOVoice()

def prep_voices():
    # Assuming voices will be used, determine which voices
    # to use and load the voices.
    global GM_VOICE, PX_VOICE, PO_VOICE
    GM_VOICE = js.getNarratorVoice()    
    d = js.document.getElementById("voicesFrom")
    idx = int(d.selectedIndex)
    #js.alert("voicesFrom: " + str(idx))
    if idx==0: # Find each agent's preferred voice.
        PX_VOICE = get_preferred_voice(PLAYERX)
        if PX_VOICE=="DEFAULT":
            PX_VOICE=get_default_PX_VOICE(PLAYERX)
            # The default might or might NOT make use of PLAYERX info in choosing.
        PO_VOICE = get_preferred_voice(PLAYERO)
        if PO_VOICE=="DEFAULT":
            PO_VOICE=get_default_PO_VOICE(PLAYERO)
    if idx==1:
        PX_VOICE=get_default_PX_VOICE(PLAYERX)        
        PO_VOICE=get_default_PO_VOICE(PLAYERO)
    else:
        PX_VOICE=get_browser_default_voice()
        PO_VOICE=get_browser_default_voice()
    js.alert("Agents' voices: "+PX_VOICE.name+", "+PO_VOICE.name)
    
def async_runGame():
    fut = ensure_future(runGame())

WAIT_TIME_AFTER_MOVES = 0.01
def set_wait_time(t):
    global WAIT_TIME_AFTER_MOVES
    WAIT_TIME_AFTER_MOVES = float(t)
     
if __name__ == '__main__':
    # Stand-alone test
    print("Starting stand-alone test of GameMaster.py")
    import TestingStage.game_types as game_types
    game_type = game_types.FIAR # Five in a Row
    print("game_types imported.")
    set_game(game_type)
    import hwang38_KInARow as h
    px = h.OurAgent()
    po = h.OurAgent(twin=True)
    set_players(px, po)
    print("Players are set.")
    print("Now to do the async_runGame")
    async_runGame()