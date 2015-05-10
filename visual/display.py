#!/usr/bin/python2
import sys
import pygame
from pygame.locals import *
from colors import *

FPS = 10

WIDTH = 1500
HEIGHT = 1500
done = False

surface = None
drawables = []

def event_handler(e):
    pass

def main(args):
    global done,surface,drawables
    pygame.init()
    fpsClock = pygame.time.Clock()

    surface = pygame.display.set_mode((WIDTH,HEIGHT))
    pygame.display.set_caption("6.834 Simulator")

    while not done:
        surface.fill(white)

        list(d(surface) for d in drawables)
        for event in pygame.event.get():
            if event.type == QUIT:
                done = True
            else:
                event_handler(event)

        fpsClock.tick(FPS)
        pygame.display.update()

    pygame.quit()
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
