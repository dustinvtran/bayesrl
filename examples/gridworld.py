"""
Prints an example of a grid world. See gridworld.py for a key to the symbols.
"""
from bayesrl.environments import GridWorld

maze = GridWorld.samples['larger']
for row in maze:
    print(row)
