''' GridWorldStateClass.py: Contains the GridWorldState class. '''

# Other imports.
from simple_rl.mdp.StateClass import State
import numpy as np

class EncodeGridWorldState(State):
    ''' Class for Grid World States '''

    def __init__(self, width, height, x, y):
        State.__init__(self, data=[x, y])
        self.x = round(x, 5)
        self.y = round(y, 5)
        self.state_dim = width * height
        self.width = width
    
    def encode(self):
        state_encoding = np.zeros(self.state_dim)
        state_encoding[(self.y-1) * self.width + self.x - 1] = 1
        return state_encoding

    def __hash__(self):
        return hash(tuple(self.data))

    def __str__(self):
        return "s: (" + str(self.x) + "," + str(self.y) + "," + str(self.encode()) + ")"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return isinstance(other, EncodeGridWorldState) and self.x == other.x and self.y == other.y
