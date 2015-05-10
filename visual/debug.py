#!/usr/bin/python2
import IPython
from threading import Thread
from grid import SuperMarket
import display
import sys

g = SuperMarket()
a1,a2,a3,a4 = g.actions
display.drawables.append(g.draw)
t = Thread(target=display.main, args=[sys.argv])
t.start()
IPython.embed()
display.done = True
t.join()
