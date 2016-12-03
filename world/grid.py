import numpy as np


class Grid:
    def __init__(self, nrows, ncols, solution=[]):
        # Initialize grid as a 2d num9py array
        self.grid = np.ndarray((nrows, ncols), dtype=object)  # the dtype is an object, not a number
        self.connect(self.grid)

        # The read head
        self.head = self.grid[(0, 0)]

        # Correct actions
        self.solution = solution
        self.a_i = 0

    def populate(self, data):
        # TODO: make this work with 2d grids as well
        for idx, val in enumerate(data):
            self.grid[0, idx].data = val

    def step(self):
        try:
            a = self.solution[self.a_i]
            self.head = self.head.__dict__[a]
            self.a_i += 1
            return self.head.data
        except:
            return None

    def reset_head(self):
        self.head = self.grid[(0, 0)]
        self.a_i = 0

    def __str__(self):
        return np.array([[c.V for c in row] for row in self.grid.tolist()]).__str__()

    def connect(self, grid):
        # Populate the grid with Cells
        for (pos, c) in np.ndenumerate(grid):
            self.grid[pos[0], pos[1]] = Cell(['l', 'r'])

        # Link cells (you can think of a grid as a 2d linked list of Cell objects)
        for (pos, c) in np.ndenumerate(grid):
            if pos[0] < grid.shape[0] - 1:
                c.u = grid[pos[0] + 1, pos[1]]
            if pos[0] > 0:
                c.d = grid[pos[0] - 1, pos[1]]
            if pos[1] > 0:
                c.l = grid[pos[0], pos[1] - 1]
            if pos[1] < grid.shape[1] - 1:
                c.r = grid[pos[0], pos[1] + 1]


class Cell:
    def __init__(self, actions):
        # Cell data
        self.data = 0

        # Add actions to dicts
        self.actions = actions
        for a in self.actions:
            self.__dict__[a] = None

        # State-action values (depends on action)
        # The state is just the current cell
        self.Q = dict.fromkeys(self.actions, 0)

        # State value
        self.V = 0

    # Return Q for given action
    def q(self, a):
        return self.Q[a]

    # Max action
    def max_action(self):
        # Take the max([of q values of [actions which don't lead to None])
        max([self.Q[a] for a in [x for x in self.actions.keys() if self.actions[x] is not None]])
