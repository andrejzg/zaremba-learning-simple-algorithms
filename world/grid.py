import numpy as np

from itertools import chain
from world.data import Data
from world.cell import Cell

from world import tasks
from world.data import Data

class Grid:
    def __init__(self, task, data, complexity=6, start_action='r'):
        self.inputs, self.actions, self.outputs, grid_size = data.task(task, complexity)
        self.nrows = grid_size[0]
        self.ncols = grid_size[1]

        self.grid = [[Cell(self.actions.chars) for _ in range(self.ncols)] for _ in range(self.nrows)]

        self.__populate()
        self.__connect()

        # Start state
        self.start = self.grid[0][0]

    def __str__(self):
        return str(self.inputs.chars)

    def __populate(self):
        for c, vec, char in zip(self.flatten(), self.inputs.vecs.flatten(), self.inputs.chars):
            c.vec = vec
            c.char = char

    def flatten(self):
        "Flatten one level of nesting"
        for c in chain.from_iterable(self.grid):
            yield c

    def __connect(self):
        # Link cells (you can think of a grid as a 2d linked list of Cell objects)
        for pos, c in np.ndenumerate(self.grid):
            if pos[0] < self.nrows - 1:
                c.u = self.grid[pos[0] + 1][pos[1]]
            if pos[0] > 0:
                c.d = self.grid[pos[0] - 1][pos[1]]
            if pos[1] > 0:
                c.l = self.grid[pos[0]][pos[1] - 1]
            if pos[1] < self.ncols - 1:
                c.r = self.grid[pos[0]][pos[1] + 1]


data = Data()
input_tape = Grid(task=tasks.reverse, data=data, complexity=6)
print(input_tape.grid[0][0].r)