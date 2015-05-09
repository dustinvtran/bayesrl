#!/usr/bin/python2
import pygame

black = pygame.Color(0,0,0)
white = pygame.Color(255,255,255)
blue  = pygame.Color(0,0,255)
green = pygame.Color(0,255,0)
red   = pygame.Color(255,0,0)
nameToColor = {
    'black' : black,
    'white' : white,
    'blue'  : blue,
    'green' : green,
    'red'   : red
}

gray = lambda fraction: (lambda c: pygame.Color(c,c,c))(int((1-fraction)*255))
