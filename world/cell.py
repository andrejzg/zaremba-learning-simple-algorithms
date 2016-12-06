class Cell:
    def __init__(self, actions):
        # Cell data
        self.data = 0

        self.actions = actions
        # Add actions to dicts
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

    def allowed_actions(self):
        return [x for x in self.actions if self.__dict__[x] is not None]

    # Max action
    def max_action(self):
        Qs = {k: self.Q[k] for k in self.allowed_actions()}
        return max(Qs, key=Qs.get)

    def take_action(self, action):
        return self.__dict__[action]
