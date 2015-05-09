#!/usr/bin/python2
from threading import Lock
import pygame
from pygame.locals import *
from colors import *
import random

class Grid(object):
    def __init__(self, height, width, aisles, robot):
        self.height = height
        self.width = width
        self.aisles = set(aisles)
        self.robot = robot
        self.l = Lock()

    def blocked(self, (r,c)):
        return (r,c) in aisles

    def set_robot(self, (r,c)):
        if self.blocked((r,c)):
            return
        with self.l:
            self.robot = (r,c)

    def dimensions(self,surface):
        pix_height = surface.get_height()
        pix_width = surface.get_width()

        row_height = int(pix_height/self.height)
        col_width = int(pix_height/self.width)

        return pix_height,pix_width,row_height,col_width

    def draw(self,surface):
        pix_height,pix_width,row_height,col_width = self.dimensions(surface)
        # Draw rows
        #
        for r in range(1,self.height):
            pygame.draw.line(surface,black,(0,r*row_height),(pix_width,r*row_height))
        # Draw columns
        #
        for c in range(1,self.width):
            pygame.draw.line(surface,black,(c*col_width,0),(c*col_width,pix_height))

        # Draw the aisles
        #
        for (r,c) in self.aisles:
            surface.fill(black, rect=(c*col_width,r*row_height,col_width,row_height))

        with self.l:
            (r,c) = self.robot
        (x,y) = int((c+0.5)*col_width),int((r+0.5)*row_height)
        radius = int(min(row_height,col_width)/2.0)
        pygame.draw.circle(surface,red,(x,y),radius,10)

class SuperMarket(Grid):
    def __init__(self):
        aisles = [
            (1,1),(2,1),(3,1),(4,1),
            (1,3),(2,3),(3,3),(4,3),
            (1,5),(2,5),(3,5),(4,5)
        ]
        width = height = 7
        possible_robot = [(0,0),(6,6)]
        robot = random.choice(possible_robot)
        possible_robot = set(possible_robot)

        super(SuperMarket,self).__init__(height,width,aisles,robot)
        self.belief = [[1./len(possible_robot) if (r,c) in possible_robot else 0.
                        for r in range(height)] for c in range(width)]

    def draw(self,surface):
        # Draw belief
        #
        with self.l:
            belief = [self.belief[r][:] for r in range(self.height)]
        pix_height,pix_width,row_height,col_width = self.dimensions(surface)
        for r in range(self.height):
            for c in range(self.width):
                surface.fill(gray(belief[r][c]),
                             rect=(c*col_width,r*row_height,col_width,row_height))
        super(SuperMarket,self).draw(surface)
