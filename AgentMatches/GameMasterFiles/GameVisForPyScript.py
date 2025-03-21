print("Loading seed file GameVisForPyScript.py")

'''GameVisForPyScript.py
    Initializes some display stuff, and then provides the
    render_state method for turning a state description
    into a Canvas-graphics rendering.

Development plan as of Nov. 16 at 11:00 AM:

1. Write a function that gets the game board dimensions N and M
and is callable from WebGameClient whenever the user changes
the game type.  Connect it to a Selection element on the page.


 Verify that Python variables hold the size of the CANVAS object.



'''

import js  # Comes from the main HTML page and the script it contains.
from pyodide.ffi import create_proxy, to_js
#from MEDIA_RESOURCES import text_items

CTX = None
CANVAS = None
#BORDERWIDTH = None
#HBWIDTH = None # Half the border width
#ACTUAL_ORIGIN_X = None
#ACTUAL_ORIGIN_Y = None
#ACTUAL_LR_CORNER_X = None
#ACTUAL_LR_CORNER_Y = None
#CANVAS_WIDTH = None
#CANVAS_HEIGHT = None
#SCALE_X = None
#SCALE_Y = None

def set_up_gui(n, m):
  '''Do any pre-session setup that might be
  needed, like computing constantss.'''
  global CTX, CANVAS #, BORDERWIDTH, HBWIDTH, \
#   ACTUAL_ORIGIN_X, ACTUAL_ORIGIN_Y,ACTUAL_LR_CORNER_X,ACTUAL_LR_CORNER_Y,\
#   CANVAS_WIDTH, CANVAS_HEIGHT, SCALE_X, SCALE_Y

  CTX = js.CTX
  CANVAS = js.CANVAS
  CANVAS.width = m*32
  CANVAS.height = n*32
  
  #Dialog = js.document.getElementById("Dialog")
  #Dialog.height = CANVAS.height
  # The above two lines are throwing an error as of Nov. 18
  # The Dialog variable has a value of None.
  # It's not clear this would equalize the heights of the
  # canvas and the dialog anyway, as there are other layout
  # related issues going on in the page.
  
#  CANVAS_WIDTH = js.CANVAS_WIDTH
#  CANVAS_HEIGHT = js.CANVAS_HEIGHT
#  BORDERWIDTH = 10 # Probably should be imported from the Problem Formulation.
#  HBWIDTH = BORDERWIDTH / 2
  # Compute usable subcanvas not taken up by the "outer" half of the
  # black border.
  # I.e., the full black border has to fit inside the canvas, but the
  # operators (and the rectangles in the state) pretend that the 
  # dimensions of the geometrical space are 1.0 by 1.0.
#  ACTUAL_ORIGIN_X = HBWIDTH
#  ACTUAL_ORIGIN_Y = HBWIDTH
#  ACTUAL_LR_CORNER_X = CANVAS_WIDTH - HBWIDTH
#  ACTUAL_LR_CORNER_Y = CANVAS_HEIGHT - HBWIDTH
#  SCALE_X = (CANVAS_WIDTH - BORDERWIDTH)
#  SCALE_Y = (CANVAS_HEIGHT - BORDERWIDTH)

  # PUT SOMETHING ON THE SCREEN... ANYTHING.
  CTX.lineWidth = 2 #BORDERWIDTH
  CTX.fillStyle = "red"
  cx = 0; cy = 0; w = CANVAS.width; h = CANVAS.height
  CTX.fillRect(cx, cy, w, h)
  CTX.strokeStyle = "black"
  CTX.strokeRect(cx, cy, w, h)
  
  #print("In GameVisForPyScript.py, finished call to set_up_gui.")
  
def cx_from_x(x):
  # Converts a state's x value to a canvas x.
  return x * SCALE_X + ACTUAL_ORIGIN_X

def cy_from_y(y):
  # Converts a state's x value to a canvas x.
  return y * SCALE_Y + ACTUAL_ORIGIN_Y

c_side = 32 # Based on the pixels dimensions of component .png images.

def render_state_canvas_graphics(s):
    CTX.clearRect(0, 0, CANVAS.width, CANVAS.height)
    # loop through the squares, drawing each one.

    board = s.board


    cy = 0
    for row in board:

        cx = 0
        for item in row:
            
             draw_image(cx, cy, item)
             cx += c_side

        cy += c_side

def draw_text(cx, cy, item):
    w = c_side; h = c_side
    CTX.fillStyle = "red"
    CTX.font = "bold 30px serif"
    #CTX.textAlign = 'left'
    CTX.textAlign = 'center'
    CTX.fillText(item, cx + w/2, cy + h/2)
    #CTX.drawText(item, cy, cy)

#    renderCommentary("It is "+who+"'s turn to move.\n")
def draw_image(cx, cy, item):
  w = c_side; h = c_side
  image = {"X": "X128", "O": "O128", " ": "gray128", "-": "black128"}[item]
  img = image+".png"
  #js.OLD_get_and_use_image(img, create_proxy(lambda img: CTX.drawImage(img, cx, cy, w, h)))
  Im = js.Image.new();
  Im.src = "GameMasterFiles/img/"+img
  Im.onload = create_proxy(lambda dummy: CTX.drawImage(Im, cx, cy, w, h))
  

def draw_box(cx, cy, item, highlight=False):
  '''Wrapper for the native HTML5 javascript call to create a
  rectangle on a canvas.  Adds support for the black border, so
  the problem formulation does not need to worry about it,
  and the render state method doesn't either.'''
  #cx, cy = cx_from_x(r.x1), cy_from_y(r.y1)
  #w, h =  cx_from_x(r.x2) - cx, cy_from_y(r.y2) - cy
  w = c_side
  h = c_side
  # For now, just draw the rectangle and forget about the border.
  #CTX.beginPath()
  CTX.lineWidth = BORDERWIDTH
  CTX.fillStyle = "gray"
  if item == "X":   CTX.fillStyle = "blue"
  if item == "O":   CTX.fillStyle = "white"
  if item == "-":   CTX.fillStyle = "black"
  CTX.fillRect(cx, cy, w, h)
  CTX.strokeStyle = "brown"
  CTX.strokeRect(cx, cy, w, h)
