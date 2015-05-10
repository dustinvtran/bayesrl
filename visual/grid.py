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
        self.actions = [(0,-1),(1,0),(0,1),(-1,0)]
        self.p_error = 0.
        self.belief = [[1. if (r,c) == robot else 0. for c in range(width)] for r in range(height)]

    def action_errors(self,action):
        return [action]

    def blocked(self, (r,c)):
        return not (0 <= r < self.height and 0 <= c < self.width) or (r,c) in self.aisles

    def set_robot(self,action):
        with self.l:
            (r,c) = self.robot
        if action in self.actions:
            self.transition_update(action)
            errors = self.action_errors(action)
            if random.random() <= self.p_error:
                action = random.choice(errors)
            (dr,dc) = action
            (nr,nc) = (r+dr,c+dc)
            if not self.blocked((nr,nc)):
                with self.l:
                    self.robot = (nr,nc)

    def transition_update(self,action):
        with self.l:
            belief = [self.belief[r][:] for r in range(self.height)]

        new_belief = [[0. for c in range(self.width)] for r in range(self.height)]
        errors = self.action_errors(action)
        for r in range(self.height):
            for c in range(self.width):
                # Correct action
                #
                dr,dc = action
                nr,nc = (r+dr,c+dc) if not self.blocked((r+dr,c+dc)) else (r,c)
                new_belief[nr][nc] += (1.0-self.p_error)*self.belief[r][c]
                # Error action
                #
                for (dr,dc) in errors:
                    nr,nc = (r+dr,c+dc) if not self.blocked((r+dr,c+dc)) else (r,c)
                    new_belief[nr][nc] += self.p_error/len(errors)*self.belief[r][c]

        Z = sum([b for r in new_belief for b in r])
        with self.l:
            self.belief = [[b/Z for b in r] for r in new_belief]

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
        aisle1 = [(1,1),(2,1),(3,1),(4,1)]
        aisle2 = [(1,3),(2,3),(3,3),(4,3)]
        aisle3 = [(1,5),(2,5),(3,5),(4,5)]
        aisles = aisle1 + aisle2 + aisle3

        width = height = 7
        possible_robot = [(0,0),(6,6)]
        robot = random.choice(possible_robot)
        possible_robot = set(possible_robot)

        super(SuperMarket,self).__init__(height,width,aisles,robot)

        self.belief = [[1./len(possible_robot) if (r,c) in possible_robot else 0.
                        for r in range(height)] for c in range(width)]
        self.actions = [(0,-1),(1,0),(0,1),(-1,0)]
        self.p_error = 0.2

        meats = ['chicken','beef','pork','turkey']
        candy = ['oreo','twix','nutella','kitkat']
        dairy = ['milk','iscream','butter','curd']
        random.shuffle(meats)
        random.shuffle(candy)
        random.shuffle(dairy)
        meat_candy_dairy = [aisle1,aisle2,aisle3]
        random.shuffle(meat_candy_dairy)
        meat_aisle,candy_aisle,dairy_aisle = meat_candy_dairy
        self.obs = dict(zip(meat_aisle,meats) + zip(candy_aisle,candy) + zip(dairy_aisle,dairy))

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

    def action_errors(self,action):
        i = self.actions.index(action)
        l = len(self.actions)
        return self.actions[(i-1)%l],self.actions[(i+1)%l]

    def observe(self):
        with self.l:
            (r,c) = self.robot
        obs = ()
        for dr,dc in [(0,-1),(1,0),(0,1),(-1,0)]:
            obs += (self.obs.get((r+dr,c+dc),None),)
        return obs
