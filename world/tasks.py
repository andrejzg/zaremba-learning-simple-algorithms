import random


def reverse(chars, complexity, char_right='r', char_left='l', no_op='-', reverse_char='.'):
    # Sequence length to complexity factor for reverse is 2 so we divide complexity
    # by two to get sequence length
    seq_len = int(complexity / 2)

    # Pick seq_len random chars
    inputs = [random.choice(chars) for _ in range(seq_len)]
    inputs.append(reverse_char)

    # Select appropriate actions
    actions = [char_right for _ in range(seq_len)]
    actions.extend([char_left for _ in range(seq_len)])

    # Construct output
    outputs = ['-' for _ in range(seq_len + 1)]
    outputs.extend(inputs[::-1][1:])

    return inputs, actions, outputs


def add():
    pass
