''' GridWorldStateClass.py: Contains the GridWorldState class. '''

# Other imports.
from simple_rl.mdp.StateClass import State
import random
import numpy as np

class ContGridWorldState(State):
    ''' Class for Grid World States '''

    def __init__(self, x, y):
        State.__init__(self, data=[x, y])
        self.x = x 
        self.y = y 

    def encode(self):
        return np.array([random.random() + self.x - 1, random.random() + self.y - 1])

    def __hash__(self):
        return hash(tuple(self.data))

    def __str__(self):
        return "s: (" + str(self.x) + "," + str(self.y) + "," + str(self.encode()) + ")"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return isinstance(other, ContGridWorldState) and self.x == other.x and self.y == other.y
