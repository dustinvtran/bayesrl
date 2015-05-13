#!/usr/bin/python2
import IPython
from threading import Thread
from grid import SuperMarket
import display
import sys
import pygame
from pygame.locals import *
import agent

held = None
def event_handler(e):
    global held
    if e.type == pygame.KEYDOWN and held is None:
        if e.key == pygame.K_a:
            g.set_robot(a1)
            held = 'a'
        elif e.key == pygame.K_s:
            g.set_robot(a2)
            held = 'b'
        elif e.key == pygame.K_d:
            g.set_robot(a3)
            held = 'd'
        elif e.key == pygame.K_w:
            g.set_robot(a4)
            held = 'w'
        if held is not None:
            print g.observe()
    elif e.type == pygame.KEYUP and held:
        if e.key == pygame.K_a and held == 'a':
            held = None
        elif e.key == pygame.K_s and held == 'b':
            held = None
        elif e.key == pygame.K_d and held == 'd':
            held = None
        elif e.key == pygame.K_w and held == 'w':
            held = None

g = SuperMarket()
a = agent.Agent(g)
class Count:
    pass
def process(every_frames):
    counter = Count()
    counter.n = 0
    counter.success = False
    def autonomous_action():
        if counter.n == 0 and len(g.targets) > 0:
            counter.success = False
            a._value_iteration()
            g.set_robot(a.next_action())
            g.observe()
        elif len(g.targets) == 0:
            if not counter.success:
                counter.success = True
                print "SUCCESS!!!!!!"
        counter.n = (counter.n+1)%every_frames
    return autonomous_action

display.event_handler = event_handler
def go():
    display.process = process(10)

a1,a2,a3,a4 = g.actions
display.drawables1.append(g.draw)
display.drawables2.append(g.draw_belief)
t = Thread(target=display.main, args=[sys.argv])
t.start()
IPython.embed()
display.done = True
t.join()
