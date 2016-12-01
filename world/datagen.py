import numpy as np


class Data():
    def __init__(self, in_chars='0123456789', classes='0123456789', no_op='-', reverse='.', actions='lrud'):
        self.chars = list(set(list(in_chars) + list(classes)))
        self.chars = self.chars + list(no_op) + list(reverse) + list(actions)
        self.classes = list(map(int, list(classes))) + [-1]
        self.in_chars = list(in_chars)

        self.char2vec = self.charvec(self.chars)

    def charvec(self, chars):
        """
        Given chars it returns a char_to_vec dictionary <char, numpy_array> of one-hot vectorized chars
        """
        char_to_ix = {ch: i for i, ch in enumerate(chars)}
        char2vec = {}

        for key, value in char_to_ix.items():
            vec = np.zeros(len(chars))
            vec[char_to_ix[key]] = 1
            char2vec[key] = vec

        return char2vec

    def task(self, task, complexity):
        inputs, actions, outputs = task(self.in_chars, complexity)

        _inputs = [self.char2vec[i] for i in inputs]
        _actions = [self.char2vec[i] for i in actions]
        _outputs = [-1 if i is '-' else i for i in outputs]
        _outputs = [np.array([int(i)]) for i in _outputs]

        return _inputs, _actions, _outputs, actions
