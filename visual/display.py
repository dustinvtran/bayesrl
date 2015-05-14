#!/usr/bin/python2
import sys
import pygame
from pygame.locals import *
from colors import *

FPS = 10

# WIDTH = 800
# HEIGHT = 500
done = False

surface = None
drawables1 = []
drawables2 = []

def event_handler(e):
    pass

def process():
    pass

def main(args):
    global done,surface,drawables
    pygame.init()
    info = pygame.display.Info()
    WIDTH = info.current_w-500
    HEIGHT = (WIDTH)/2
    fpsClock = pygame.time.Clock()

    master_surface = pygame.display.set_mode((WIDTH+100,HEIGHT))
    pygame.display.set_caption("6.834 Simulator")
    surface1 = pygame.Surface(((WIDTH/2),HEIGHT))
    surface2 = pygame.Surface(((WIDTH/2),HEIGHT))

    while not done:
        process()
        master_surface.fill(black)
        surface1.fill(white)
        surface2.fill(white)

        list(d(surface1) for d in drawables1)
        list(d(surface2) for d in drawables2)
        for event in pygame.event.get():
            if event.type == QUIT:
                done = True
            else:
                event_handler(event)

        master_surface.blit(surface1,(0,0))
        master_surface.blit(surface2,(WIDTH/2+100,0))
        fpsClock.tick(FPS)
        pygame.display.flip()

    pygame.quit()
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
