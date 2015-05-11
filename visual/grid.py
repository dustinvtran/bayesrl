#!/usr/bin/python2
from threading import Lock
import pygame
from pygame.locals import *
from colors import *
import random
import IPython

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
            new_belief = self.transition_update(self.belief,action)
            with self.l:
                self.belief = new_belief
            errors = self.action_errors(action)
            if random.random() <= self.p_error:
                action = random.choice(errors)
            (dr,dc) = action
            (nr,nc) = (r+dr,c+dc)
            if not self.blocked((nr,nc)):
                with self.l:
                    self.robot = (nr,nc)

    def transition_probs(self,state,action):
	t_probs = {}
	r, c = state
        errors = self.action_errors(action)
	dr,dc = action
	new_state = (r+dr,c+dc) if not self.blocked((r+dr,c+dc)) else (r,c)
	t_probs[new_state] = 1. - self.p_error
	for (dr,dc) in errors:
	    new_state = (r+dr,c+dc) if not self.blocked((r+dr,c+dc)) else (r,c)
	    if new_state in t_probs:
		t_probs[new_state] += self.p_error/len(errors)
	    else:
		t_probs[new_state] = self.p_error/len(errors)
		
        return t_probs

    def transition_update(self,belief,action):
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
        return new_belief

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
        self.aisle1 = [(1,1),(2,1),(3,1),(4,1)]
        self.aisle2 = [(1,3),(2,3),(3,3),(4,3)]
        self.aisle3 = [(1,5),(2,5),(3,5),(4,5)]
        self.aisles = [self.aisle1,self.aisle2,self.aisle3]
        aisles = self.aisle1 + self.aisle2 + self.aisle3

        width = height = 7
        possible_robot = [(0,0),(6,6)]
        robot = random.choice(possible_robot)
        possible_robot = set(possible_robot)

        super(SuperMarket,self).__init__(height,width,aisles,robot)

        self.belief = [[1./len(possible_robot) if (r,c) in possible_robot else 0.
                        for r in range(height)] for c in range(width)]
        self.actions = [(0,-1),(1,0),(0,1),(-1,0)]
        self.p_error = 0.2

        self.targets = set(['oreo','iscream','milk'])
        self.meats = ['chicken','beef','pork','turkey']
        self.candy = ['oreo','twix','nutella','kitkat']
        self.dairy = ['milk','iscream','butter','curd']
        random.shuffle(self.meats)
        random.shuffle(self.candy)
        random.shuffle(self.dairy)
        meat_candy_dairy = [self.aisle1,self.aisle2,self.aisle3]
        random.shuffle(meat_candy_dairy)
        meat_aisle,candy_aisle,dairy_aisle = meat_candy_dairy

        self.obs = dict(zip(meat_aisle,self.meats) + zip(candy_aisle,self.candy) + zip(dairy_aisle,self.dairy))

        # Aisle belief state
        #
        self.aisles_belief = {
            1: {'meats': 1./3., 'candy': 1./3., 'dairy': 1./3.},
            2: {'meats': 1./3., 'candy': 1./3., 'dairy': 1./3.},
            3: {'meats': 1./3., 'candy': 1./3., 'dairy': 1./3.}
        }

        # Inner aisle belief state
        #
        meat_inner = lambda: dict((m,1./len(self.meats)) for m in self.meats)
        candy_inner = lambda: dict((c,1./len(self.candy)) for c in self.candy)
        dairy_inner = lambda: dict((d,1./len(self.dairy)) for d in self.dairy)
        self.content_belief = {
            'meats': dict(enumerate([meat_inner() for _ in self.meats])),
            'candy': dict(enumerate([candy_inner() for _ in self.candy])),
            'dairy': dict(enumerate([dairy_inner() for _ in self.dairy]))
        }

    def cell_to_aisle(self,(r,c)):
        return (1,self.aisle1.index((r,c))) if (r,c) in self.aisle1 else \
            (2,self.aisle2.index((r,c))) if (r,c) in self.aisle2 else \
            (3,self.aisle3.index((r,c))) if (r,c) in self.aisle3 else \
            None

    def category(self, product):
        return 'meats' if product in self.meats else \
            'candy' if product in self.candy else \
            'dairy' if product in self.dairy else \
            None

    def draw(self,surface):
        # Draw belief
        #
        with self.l:
            belief = [r[:] for r in self.belief]
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
        self.observation_update(obs)
        list(self.targets.discard(o) for o in obs)
        return obs

    EPSILON = 1e-7
    def observation_update(self, observation):
        with self.l:
            belief = [r[:] for r in self.belief]

        # Update position belief
        obs_cells = [(obs,(r+dr,c+dc),(r,c))
                     for r in range(self.height)
                     for c in range(self.width)
                     for (obs,(dr,dc)) in zip(observation, [(0,-1),(1,0),(0,1),(-1,0)])
                     if belief[r][c] > self.EPSILON
        ] # (obs,(row,col),parent)
        for (obs,neigh,(r,c)) in obs_cells:
            if self.cell_to_aisle(neigh) is None:
                if obs is not None:
                    belief[r][c] = 0
            else:
                if obs is None:
                    belief[r][c] = 0
                else:
                    aisle,pos = self.cell_to_aisle(neigh)
                    cat = self.category(obs)
                    belief[r][c] *= self.aisles_belief[aisle][cat] * self.content_belief[cat][pos][obs]
        Z = sum(b for r in belief for b in r)
        belief = [[b/Z for b in r] for r in belief]
        with self.l:
            self.belief = belief
        if not all(o is None for o in observation):
            obs_cells = [(obs,(r+dr,c+dc),belief[r][c],(r,c))
                         for r in range(self.height)
                         for c in range(self.width)
                         for (obs,(dr,dc)) in zip(observation, [(0,-1),(1,0),(0,1),(-1,0)])
                         if obs is not None
                     ] # (obs,(row,col),prob,parent)
            self.observation_world_update(obs_cells)

    def observation_world_update(self,obs_cells):
        with self.l:
            belief = [r[:] for r in self.belief]
        # Update world belief
        op = {}
        ac = {}
        for (obs,neigh,prob,(r,c)) in obs_cells:
            if self.cell_to_aisle(neigh) is None:
                continue
            aisle,pos = self.cell_to_aisle(neigh)
            cat = self.category(obs)

            if obs not in op:
                op[obs] = {}
            op[obs][pos] = op[obs].get(pos,0)+prob

            if aisle not in ac:
                ac[aisle] = {}
            ac[aisle][cat] = ac[aisle].get(cat,0)+prob

        # Update aisle beliefs
        #
        ab = self.aisles_belief
        for a in ac:
            for cat in ac[a]:
                prob = ac[a][cat]
                Z1 = ab[a][cat]
                Z2 = 1-Z1
                if Z1 == 0 or Z2 == 0:
                    continue
                for c in ab[a]:
                    ab[a][c] *= prob/Z1 if (c==cat) else (1.-prob)/Z2

        # Update aisle position beliefs
        #
        for obs in op:
            cat = self.category(obs)
            cb = self.content_belief[cat]
            for p in cb:
                prob = op[obs][p]
                Z1 = cb[p][obs]
                Z2 = 1-Z1
                if Z1 == 0 or Z2 == 0:
                    continue
                for o in cb[p]:
                    cb[p][o] *= prob/Z1 if (o==obs) else (1.-prob)/Z2
