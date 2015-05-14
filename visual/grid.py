#!/usr/bin/python2
from threading import Lock
from math import *
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
        for r in range(self.height):
            for c in range(self.width):
                t_probs = self.transition_probs((r,c),action)
                for (nr,nc) in t_probs:
                    new_belief[nr][nc] += t_probs[(nr,nc)]*self.belief[r][c]
        return new_belief

    def dimensions(self,surface):
        pix_height = surface.get_height()
        pix_width = surface.get_width()

        row_height = int(pix_height/self.height)
        col_width = int(pix_width/self.width)

        return pix_height,pix_width,row_height,col_width

    def draw(self,surface,robot=True):
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

        if robot:
            with self.l:
                (r,c) = self.robot
            (x,y) = int((c+0.5)*col_width),int((r+0.5)*row_height)
            radius = int(min(row_height,col_width)/2.0)
            pygame.draw.circle(surface,red,(x,y),radius,10)

class SuperMarket(Grid):
    def __init__(self):
        self.aisles_content = {
            'meats': ['chicken','beef','pork','turkey'],
            'candy': ['oreo','twix','nutella','kitkat'],
            'dairy': ['milk','iscream','butter','curd'],
            'drink': ['water','juice','soda','smoothi'],
            'grain': ['rice','flour','barley', 'beans'],
            'pasta': ['penne','fusilli','farfalle','lasagna']
        }

        self.aisles_list = [
            [(i+1,2*n+1) for i in range(len(self.aisles_content.values()[0]))]
            for n in range(len(self.aisles_content))
        ]
        aisles = [cell for a in self.aisles_list for cell in a]

        width = len(self.aisles_list * 2) + 1
        height = len(self.aisles_list[0])+3
        possible_robot = [(0,0),(height-1,width-1)]
        robot = random.choice(possible_robot)
        possible_robot = set(possible_robot)

        super(SuperMarket,self).__init__(height,width,aisles,robot)

        self.belief = [[1./len(possible_robot) if (r,c) in possible_robot else 0.
                        for c in range(width)] for r in range(height)]
        self.actions = [(0,-1),(1,0),(0,1),(-1,0)]
        self.p_error = 0.05

        self.targets = set(['oreo','iscream','milk'])

        for c in self.aisles_content:
            random.shuffle(self.aisles_content[c])
        aisles_order = self.aisles_list[:]
        random.shuffle(aisles_order)
        self.obs = {}
        for prods,aisle in zip(self.aisles_content.values(),aisles_order):
            self.obs.update(zip(aisle,prods))

        # Aisle belief state
        #
        self.aisles_belief = {}
        aisles_types = self.aisles_content.keys()
        for a in range(len(self.aisles_list)):
            self.aisles_belief[a+1] = dict(zip(
                aisles_types,
                [1./len(aisles_types)]*len(aisles_types)))

        # Inner aisle belief state
        #
        inner = lambda prods: dict((p,1./len(prods)) for p in prods)
        self.content_belief = dict(
            (cat,dict(enumerate([inner(prods) for _ in prods])))
            for cat,prods in self.aisles_content.items()
        )

        self.images = {
            'meats'   : pygame.image.load("images/meats.jpg"),
            'candy'   : pygame.image.load("images/candy.gif"),
            'dairy'   : pygame.image.load("images/dairy.jpg"),
            'drink'   : pygame.image.load("images/drink.jpg"),
            'grain'   : pygame.image.load("images/grain.jpg"),
            'pasta'   : pygame.image.load("images/pasta.jpg"),

            'chicken' : pygame.image.load("images/chicken.jpg"),
            'pork'    : pygame.image.load("images/pork.jpg"),
            'turkey'  : pygame.image.load("images/turkey.gif"),
            'beef'    : pygame.image.load("images/beef.jpg"),

            'oreo'    : pygame.image.load("images/oreo.jpg"),
            'twix'    : pygame.image.load("images/twix.jpg"),
            'nutella' : pygame.image.load("images/nutella.jpg"),
            'kitkat'  : pygame.image.load("images/kitkat.jpg"),

            'milk'    : pygame.image.load("images/milk.jpg"),
            'curd'    : pygame.image.load("images/curd.jpg"),
            'iscream' : pygame.image.load("images/iscream.jpg"),
            'butter'  : pygame.image.load("images/butter.jpg"),

            'water'   : pygame.image.load("images/water.jpg"),
            'juice'   : pygame.image.load("images/juice.jpg"),
            'soda'    : pygame.image.load("images/soda.jpg"),
            'smoothi' : pygame.image.load("images/smoothi.jpg"),

            'rice'    : pygame.image.load("images/rice.jpg"),
            'flour'   : pygame.image.load("images/flour.jpg"),
            'barley'  : pygame.image.load("images/barley.jpg"),
            'beans'   : pygame.image.load("images/beans.jpg"),

            'penne'   : pygame.image.load("images/penne.jpg"),
            'fusilli' : pygame.image.load("images/fusilli.jpg"),
            'farfalle': pygame.image.load("images/farfalle.jpg"),
            'lasagna' : pygame.image.load("images/lasagna.jpg")
        }

    def cell_to_aisle(self,(r,c)):
        for i in range(len(self.aisles_list)):
            if (r,c) in self.aisles_list[i]:
                return (i+1,self.aisles_list[i].index((r,c)))
        return None

    def category(self, product):
        for c in self.aisles_content:
            if product in self.aisles_content[c]:
                return c
        return None

    transformed = False
    def draw(self,surface):
        pix_height,pix_width,row_height,col_width = self.dimensions(surface)
        super(SuperMarket,self).draw(surface)
        if not self.transformed:
            for prod in self.images:
                img = self.images[prod]
                self.images[prod] = pygame.transform.scale(img.convert(),(col_width,row_height))
            self.transformed = True
        for (r,c) in self.aisles:
            prod = self.obs[(r,c)]
            img = self.images.get(prod,None)
            if img is not None:
                surface.blit(img, dest=(c*col_width,r*row_height))

    def draw_belief(self,surface):
        with self.l:
            belief = [r[:] for r in self.belief]
        pix_height,pix_width,row_height,col_width = self.dimensions(surface)
        for r in range(self.height):
            for c in range(self.width):
                logB = log(belief[r][c]) if belief[r][c] != 0 else -12
                surface.fill(gray(max((logB+12)/12.0,0)),
                             rect=(c*col_width,r*row_height,col_width,row_height))

        super(SuperMarket,self).draw(surface,False)

        if not self.transformed:
            for prod in self.images:
                img = self.images[prod]
                self.images[prod] = pygame.transform.scale(img.convert(),(col_width,row_height))
            self.transformed = True

        for (r,c) in self.aisles:
            # Are we more than 50% certain about any product here?
            found = False
            a,p = self.cell_to_aisle((r,c))
            for cat in self.aisles_belief[a]:
                prob_cat = self.aisles_belief[a][cat]
                if prob_cat > 0.5:
                    img = self.images.get(cat,None)
                    found = True
                    for prod in self.content_belief[cat][p]:
                        prob = prob_cat*\
                               self.content_belief[cat][p][prod]
                        if prob > 0.5:
                            img = self.images.get(prod,None)
                            break
                    break
            if not found:
                continue
            surface.blit(img, dest=(c*col_width,r*row_height))

    def action_errors(self,action):
        i = self.actions.index(action)
        l = len(self.actions)
        return self.actions[(i-1)%l],self.actions[(i+1)%l],(0,0)

    def observe(self):
        with self.l:
            (r,c) = self.robot
        obs = ()
        for dr,dc in [(0,-1),(1,0),(0,1),(-1,0)]:
            obs += (self.obs.get((r+dr,c+dc),None),)
        self.observation_update(obs)
        list(self.targets.discard(o) for o in obs)
        return obs

    def observation_update(self, observation):
        with self.l:
            belief = [r[:] for r in self.belief]

        # Update position belief
        obs_cells = [(obs,(r+dr,c+dc),(r,c))
                     for r in range(self.height)
                     for c in range(self.width)
                     for (obs,(dr,dc)) in zip(observation, [(0,-1),(1,0),(0,1),(-1,0)])
                     if belief[r][c] != 0
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
