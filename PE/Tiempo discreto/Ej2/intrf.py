#
#  Project: Simple interface for ANSI terminals
#  File:    intrf.py
#  Vers.    1.0
#  Date:    06/26/2020
#
#
#  This file contains functions and classes to implement simple
#  interfaces on a text-based ANSI compliant terminal. It defines
#  functions to manipulate the terminal and classes that define
#  interface elements.
#
import sys

#
# Label positions for the slidebar
#
ONTOP = 1
LEFT = 2
NONE = 3

#
#  Foreground and background colors, values from 0 to 9
#
_fore = 0
_back = 9


#
# Size of the window (works only if resized)
#
_rows = 0
_cols = 0

#
# Sets the new foreground color
#
def set_fore(fg):
    global _fore
    if fg >=0 and fg <=9:
        _fore = fg
    return

#
# Returns the value of the current foreground color
#
def fore():
    global _fore
    return _fore

#
# Sets the new foreground color
#
def set_back(bg):
    global _back
    if bg >=0 and bg <=9:
        _back = bg
    return

#
# Returns the value of the current background color
#
def back():
    global _back
    return _back


#
# Clears the screen, setting it into the foreground and background
# colors
#
def clear():
    global _fore, _back
    sys.stdout.write("%c[%d;%dm" % (27, 30 + _fore, 40 + _back))
    sys.stdout.flush()
    sys.stdout.write("%c[2J" % 27)
    sys.stdout.flush()

#
# Resizes the current window        
#
def resize(rows, cols):
    global _rows, _cols
    _rows = rows
    _cols = cols
    sys.stdout.write("%c[8;%d;%dt" % (27, rows, cols))
    sys.stdout.flush()

#
#  Paints the window empty in the current background color. Works only
#  if the window has been resized
#
def repaint():
    global _rows, _cols
    global _fore, _back
    if _rows <= 0 or _cols <= 0:
        return

    sys.stdout.write("%c[%d;%dm" % (27, 30 + _fore, 40 + _back))
    buf = _cols*" "
    for r in range(_rows):
        sys.stdout.write("%c[%d;%dH" % (27, r, 0))
        sys.stdout.write(buf)
        
    sys.stdout.flush()
    return


#
# Moves the cursor to a given location of the screen
# (0,0) is the upper left corner of the screen
#
def move_to(row, col):
    sys.stdout.write("%c[%d;%dH" % (27, row+1, col+1))

#
# Prints a string at a given location of the screen
# (0,0) is the upper left corner of the screen
#
def print_at(row, col, s):
    sys.stdout.write("%c[%d;%dm" % (27, 30 + _fore, 40 + _back))
    sys.stdout.write("%c[%d;%dH" % (27, row+1, col+1))
    sys.stdout.write(s)        
    sys.stdout.flush()
    return



#
#  Class slidebar
#
#  A slidebar is an indicator of a quantity displayed on screen as a
#  rectangle that will be partially filled depending on the value that 
#  we want to indicate. The bar has an optional label that can be placed 
#  on top of it or at its right
#
#
#        <label on top>
#        +---------------------------------+
#        |                                 | <label on the left>
#  (r,c) +---------------------------------+
#        ^                               ^
#        +--------    n    --------------+
#
#
#  Constructor parameters:
#  vmin:   minimal value that is dosplayed (corresponds to the empty bar)  
#  vmax:   maximal value displayes (correspond sto the full bar)
#  r:      row of the lower left corner of the bar
#  c:      column of the lower left corner of the bar
#  n:      size in character of the inside of the bar
#  lpos:   position of the label: intrf.ONTOP, intrf.LEFT, or intrf.NONE
#  label:  optional label (default: "")
#
#  Note that the bar occupies on screen 3 rows and n+2 columns plus the label.
#  The row value must be at least equal to 2, an the column to 0 No attempt is made 
#  to check that the bar actually fits the screen
#
class slidebar:
    n = 0
    r = 0
    c = 0
    lpos = 0
    label = ""
    vmin = 0.0
    vmax = 0.0
    lstfill = 0
    labelrow = 0
    labelcol = 0


    def _draw_contour(self):
        side = self.n*"-"
        fill = self.n*" "
        rw = self.r-2
        sys.stdout.write("%c[%d;%dH" % (27, rw, self.c))
        sys.stdout.write("+%s+" % side)
        sys.stdout.write("%c[%d;%dH" % (27, rw+1, self.c))
        sys.stdout.write("|%s|" % fill)
        sys.stdout.write("%c[%d;%dH" % (27, rw+2, self.c))
        sys.stdout.write("+%s+" % side)

        if self.lpos != NONE:
            sys.stdout.write("%c[%d;%dH" % (27, self.labelrow, self.labelcol))
            sys.stdout.write("%s" % self.label)

        return


    def __init__(self, vmin, vmax, r, c, n, lpos=NONE, label=""):
        self.vmin = vmin
        self.vmax = vmax
        self.r = r
        self.c = c
        self.n = n
        self.lpos = lpos
        self.label = label
        self.lastfill = 0
        if self.lpos == ONTOP:
            self.labelrow = self.r-3
            self.labelcol = self.c
        elif self.lpos == LEFT:
            self.labelrow = self.r-1
            self.labelcol = self.c+self.n+4


        self._draw_contour()

    def fill(self, val):
        nfill = int(self.n*(val-self.vmin)/(self.vmax-self.vmin))
        if nfill < 0:
            nfill = 0
        if nfill >= self.n:
            nfill = self.n
        if self.lastfill > 0:
            blk = self.lastfill*" "
            sys.stdout.write("%c[%d;%dm" % (27, 30 + _fore, 40 + _back))
            sys.stdout.write("%c[%d;%dH" % (27, self.r-1, self.c+1))
            sys.stdout.write(blk)
        self.lastfill = nfill
        if self.lastfill > 0:
            blk = self.lastfill*" "
            sys.stdout.write("%c[%d;%dm" % (27, 30 + _back, 40 + _fore))
            sys.stdout.write("%c[%d;%dH" % (27, self.r-1, self.c+1))
            sys.stdout.write(blk)
            sys.stdout.write("%c[%d;%dm" % (27, 30 + _fore, 40 + _back))
        sys.stdout.flush()
        return

    #
    # Changes the label of the bar. The label is left in the same
    # position and displayed immediately. The function makes sure that
    # the old one is erasedm even if the new one is shorter
    #
    def setlabel(self, label):
        if self.lpos != NONE:
            dud = len(self.label)*" "
            sys.stdout.write("%c[%d;%dH" % (27, self.labelrow, self.labelcol))
            sys.stdout.write("%s" % dud)
            self.label = label
            sys.stdout.write("%c[%d;%dH" % (27, self.labelrow, self.labelcol))
            sys.stdout.write("%s" % label)
