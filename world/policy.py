import random


def e_greedy(cell, e):
    if random.uniform(0, 1) <= e:
        a = random.choice(cell.allowed_actions())
    else:
        a = greedy(cell)
    return a


def greedy(cell):
    Qs = {k: cell.Q[k] for k in cell.allowed_actions()}
    return max(Qs, key=Qs.get)
