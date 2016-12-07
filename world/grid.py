import numpy as np

from world.data import Data
from world.cell import Cell


class Grid:
    def __init__(self, task, complexity=6, start_action='r'):
        data = Data()
        self.inputs, self.actions, self.outputs, grid_size = data.task(task, complexity)
        self.grid = np.ndarray((grid_size[0], grid_size[1]), dtype=object)
        self.__instatiate()
        self.__populate()
        self.__connect()

        # Start state
        self.start = Cell(start_action)
        self.start.r = self.grid.item(0)

    def __str__(self):
        return str(self.inputs.chars)

    def __instatiate(self):
        for pos, _ in np.ndenumerate(self.grid):
            self.grid[pos[0], pos[1]] = Cell(self.actions.chars)

    def __populate(self):
        for c, vec, char in zip(self.grid.flatten(), self.inputs.vecs.flatten(), self.inputs.chars):
            c.vec = vec
            c.char = char

    def __connect(self):
        # Link cells (you can think of a grid as a 2d linked list of Cell objects)
        for pos, c in np.ndenumerate(self.grid):
            if pos[0] < self.grid.shape[0] - 1:
                c.u = self.grid[pos[0] + 1, pos[1]]
            if pos[0] > 0:
                c.d = self.grid[pos[0] - 1, pos[1]]
            if pos[1] > 0:
                c.l = self.grid[pos[0], pos[1] - 1]
            if pos[1] < self.grid.shape[1] - 1:
                c.r = self.grid[pos[0], pos[1] + 1]
