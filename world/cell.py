class Cell:
    def __init__(self, actions, vec=0, char=0):
        # Cell data
        self.vec = vec  # the data vector the cell holds
        self.char = char  # the corresponding character

        # Cell actions
        self.actions = actions

        # Add actions to __dict__
        for a in self.actions:
            self.__dict__[a] = None

        # State-action values (depends on action)
        # The state is just the current cell
        self.Q = dict.fromkeys(self.actions, 0)

        # State value
        self.V = 0

        # Eligibility traces
        self.E = dict.fromkeys(self.actions, 0)

    # Return Q for given action
    def q(self, a):
        return self.Q[a]

    def allowed_actions(self):
        return [x for x in self.actions if self.__dict__[x] is not None]

    def take_action(self, action):
        return self.__dict__[action]
