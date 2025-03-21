'''WebGameClient.py

NEW VERSION Nov 17.

"Client" for running web game master.

Works together with index.html (the web page),
  GameMaster.py -- the program that runs the games,
    but knows almost nothing about the Web context and
    could run without being on the web..
  GameVisForPyScript.py (which handles rendering the
     states, including animations, and rendering the
     utterances, including possible speech synthesis;
     this is web-browser intensive), 
  wgm.css (which has some styling specific to the web game master),

  and other various resources and possibly JS utilities.

'''

from pyodide.ffi import create_proxy, to_js
import js  # Makes top-level Javascript fns callable from Python.
from js import window, document

import sys
import time
START_TIME = time.time()
USE_VOICES = False;

# Use this to update the transcript of the game on the screen:
def add_to_dialog(stuff):
  Dialog = document.getElementById('dialog')
  Dialog.innerHTML += stuff + "\n"

js.console.log("In WebGameClient.py about to import GameMaster")
import GameMaster as gm
js.console.log("import succeeded.")

# OVER-WRITE THE DEFINITION of GameMaster.renderCommentary with
# the one here, so rendering will happen on the web page rather than
# the console.
gm.add_to_history = add_to_dialog

# import the various games available
from game_types import Game_Type, TTT, FIAR, Cassini

AGENT_LIST = []

GAME_INFO = None
def show_game_info():
  global GAME_INFO, AGENT_LIST
  if AGENT_LIST==[]:
     import RandomPlayer as Randy
     import hwang38_KInARow as Mark_Lee
     import kohei28_KInARow as N_System
     import nate235_KInARow as Calculating_Nate
     import zhengkun_KInARow as Bruh
     import cadetank_KInARow as Kevin_the_minion
     import Notorious_JAH_KInARow as Notorious_Jah
     AGENT_LIST.append(Randy.OurAgent())
     AGENT_LIST.append(Mark_Lee.OurAgent())
     AGENT_LIST.append(N_System.OurAgent())
     AGENT_LIST.append(Calculating_Nate.OurAgent())
     AGENT_LIST.append(Bruh.OurAgent(twin=True))
     AGENT_LIST.append(Kevin_the_minion.OurAgent())
     AGENT_LIST.append(Mark_Lee.OurAgent(twin=True))
     AGENT_LIST.append(Notorious_Jah.OurAgent())
     # When available, put more agents here, AND in index.html's PY-CONFIG,
     # and import them here, too, like Hobgoblin.
     #print("AGENT_LIST:", AGENT_LIST)

  populate_agent_choice("playerX", AGENT_LIST, 0)
  populate_agent_choice("playerO", AGENT_LIST, 1)

def populate_agent_choice(selectorId, AGENT_LIST, selected):
  selector = document.getElementById(selectorId)
  for i, agent in enumerate(AGENT_LIST):
    if i==selected:
      is_sel = " selected"
    else:
      is_sel = ""
    selector.innerHTML += "<option" + is_sel + ">"+agent.long_name+ "</option>\n"

def show_item(item_name, GAME_INFO):
  js.console.log("Getting DOM_element in show_item")
  DOM_element = document.getElementById(item_name)
  js.console.log("  It's "+str(DOM_element))
  DOM_element.innerHTML = GAME_INFO[item_name]

def game_selected(event):
  game_selector = document.getElementById("game-select")
  selectedIndex = game_selector.selectedIndex
  if selectedIndex == 1: game = TTT
  elif selectedIndex == 2: game = FIAR
  elif selectedIndex == 3: game = Cassini
  else:
    js.alert("Unknown game selected: " + str(selectedIndex))
    return
  gm.set_game(game)
  # Redo the display of the initial state:
  vis.set_up_gui(game.n, game.m)
  # Clear the dialog in prep for a new game:
  Dialog = document.getElementById('dialog')
  Dialog.innerHTML =  ""
  
def wait_time_selected(event):
  time_selector = document.getElementById("wait-after-moves")
  selectedIndex = time_selector.selectedIndex
  id = "T"+str(selectedIndex)
  opt = document.getElementById(id)
  wait_time = opt.value
  gm.set_wait_time(wait_time)
  #js.alert("Wait time after each move is now "+opt.text)

def showVoicesHelp(e):
  js.alert("If using voices on this website for the first time, enable access to audio for this website in your browser.  In Chrome, right click on the icon to the left of the URL bar and select 'site settings'. Scroll down to 'Sound' and select 'Allow'.. \n For help selecting voices for your agent, see https://codepen.io/matt-west/pen/DpmMgE")

def updateUseOfVoices(e):
  global USE_VOICES
  elt = js.document.getElementById("useVoices")
  if elt.selectedIndex == 1:
    USE_VOICES = True
    js.startVoices()
  else:
    USE_VOICES = False
    js.alert("Voices now off")
  gm.USE_VOICES = USE_VOICES # share info with the Game Master
  
def skipUtterance(e):
  if not USE_VOICES: return
  #  js.alert("Called skipUtterance.")

def changeSpeakingRate(e):
  d = document.getElementById("rate")
  newRate = d.value
  #  js.alert("SpeakingRate new value is "+str(newRate))

def newChoiceOfVoices(e):
  d = document.getElementById("voicesFrom")
  idx = d.selectedIndex
  # js.alert("Called newChoiceofVoices:" + str(idx))
    
def set_up_handlers():
  show_game_info()
  d = document.getElementById("game-select")
  d.addEventListener("change", create_proxy(game_selected))

  d = document.getElementById("start")
  d.addEventListener("click", create_proxy(start_game))

  d = document.getElementById("wait-after-moves")
  d.addEventListener("change", create_proxy(wait_time_selected))
  
  d = document.getElementById("voicesHelp")
  d.addEventListener("click", create_proxy(showVoicesHelp))

  d = document.getElementById("useVoices")
  d.addEventListener("change", create_proxy(updateUseOfVoices))

#  d = document.getElementById("skipUtterance")
#  d.addEventListener("click", create_proxy(skipUtterance))

#  d = document.getElementById("rate")
#  d.addEventListener("change", create_proxy(changeSpeakingRate))

  d = document.getElementById("voicesFrom")
  d.addEventListener("change", create_proxy(newChoiceOfVoices))

  js.console.log("user interface has been set up.")

import time
SESSION_START_TIME = time.localtime()
SESSION_START_SECONDS = time.time()

def elapsed_seconds():
  return time.time() - SESSION_START_SECONDS

def elapsed_minutes():
  return elapsed_seconds() / 60.0

CC_AREA = document.getElementById("CC")

def renderCommentary(text):
  CC_AREA.innerHTML = text
  add_to_dialog(text)

def renderState(s):
  vis.render_state_canvas_graphics(s)
  add_to_dialog(str(s))
  
import GameVisForPyScript as vis
gm.set_game(FIAR) # In case it has not been selected.
vis.set_up_gui(gm.N, gm.M)
gm.printState = renderState
gm.renderCommentary = renderCommentary

def start_game(event):
  Dialog = document.getElementById('dialog')
  Dialog.innerHTML =  ""
  selector = document.getElementById("playerX")
  i = selector.selectedIndex
  selector = document.getElementById("playerO")
  j = selector.selectedIndex
  px = AGENT_LIST[i]
  po = AGENT_LIST[j]
  gm.set_players(px, po)
  gm.async_runGame()

set_up_handlers()
