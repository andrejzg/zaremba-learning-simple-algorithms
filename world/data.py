import numpy as np
from collections import namedtuple


class Data:
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

        vec_inputs = np.array([self.char2vec[i] for i in inputs])
        vec_actions = np.array([self.char2vec[i] for i in actions])
        vec_outputs = [-1 if i is '-' else i for i in outputs]
        vec_outputs = [np.array([int(i)]) for i in vec_outputs]

        Datum = namedtuple('Datum', 'chars vecs')

        i = Datum(inputs, vec_inputs)
        a = Datum(actions, vec_actions)
        o = Datum(outputs, vec_outputs)

        grid_size = (1, complexity)

        return i, a, o, grid_size
