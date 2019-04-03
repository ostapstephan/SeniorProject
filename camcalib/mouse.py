import sys
import os

from time import sleep


import threading

# uses the package python-xlib
# from http://snipplr.com/view/19188/mouseposition-on-linux-via-xlib/
# or: sudo apt-get install python-xlib
from Xlib import display


old_stdout = sys.stdout
sys.stdout = open(os.devnull, 'w')

def mousepos():
    """mousepos() --> (x, y) get the mouse coordinates on the screen (linux, Xlib)."""
    data = display.Display().screen().root.query_pointer()._data
    return data["root_x"], data["root_y"]



for i in range(10):
    print(mousepos())
    sleep(1)

