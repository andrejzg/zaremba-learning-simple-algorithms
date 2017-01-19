import numpy as np
import world.policy as policy

from sklearn import neural_network

from world.grid import Grid
from world.data import Data
from world import tasks


def main():
    ########################################################################
    #                                   TRAIN
    ########################################################################

    data = Data()  # standard data 0-9 input/output '-' no-op and '.' reverse symbol 'udlr' actions
    epochs = 10
    model = neural_network.MLPClassifier(activation='logistic', hidden_layer_sizes=(200,))
    alpha = 0.02
    gamma = 0.98
    lmbda = 1

    for epoch in range(epochs):
        complexity = (epoch + 1) * 4 + 2
        acc = 0
        step = 1

        model.partial_fit(data.char2vec['r'], np.array([-1]), classes=data.classes)
        print('\n\n======= EPOCH\t{}/{}\tcomplexity:{} ======='.format(epoch + 1, epochs, complexity))

        while acc < 1.0:
            # TODO: change to a.vec
            # new task
            input_tape = Grid(task=tasks.reverse, data=data, complexity=complexity)

            # initial state, action
            s = input_tape.start
            a = 'r'     # a here is a dummy action which we need to create x
            a_vec = data.char2vec[a]

            # steps remaining to complete task
            steps_left = complexity

            print('\n\ncomplexity: {} solution: {}'.format(complexity, input_tape.outputs.chars))

            steps_left = complexity

            while True:
                a_vec = data.char2vec[a]
                x = (s.vec + a_vec).reshape(1, -1)  # combine input + prev action into input x
                y = input_tape.outputs.vecs.pop(0)  # expected output y
                    # predicted y using model
                y_pred = model.predict(x)
                model.partial_fit(x, y, classes=data.classes)  # fit model

                # reward signal
                if y == y_pred:
                    reward = 1
                    if y != -1:
                        print('YESSS!!!!!!')
                    print('CORRECT! {}\t\ty: {}\ty_pred: {}'.format(a, y[0], y_pred[0]))
                else:
                    print('WRONG! {}\t\ty: {}\ty_pred: {}'.format(a, y[0], y_pred[0]))
                    break

                s_new = s.take_action(a)                # take action a to get s'
                a_new = policy.e_greedy(s_new, 0.2)     # choose a' from s' using e-greedy policy
                a_star = policy.greedy(s_new)           # a* = argmax_b Q(s', b)
                if s.Q[a_new]/steps_left == s.Q[a_star]:           # if a' ties for max, then a* = a'
                    a_star = a_new

                delta = reward + gamma * s_new.Q[a_star] - s.Q[a]   # delta (go over this again)
                s.E[a] += 1                                         # update eligibility

                for _s in input_tape.flatten():
                    for _a in ['r', 'l']:
                        _s.Q[a] += alpha * delta * _s.E[_a] / steps_left
                        if a_new == a_star:
                            _s.E[_a] *= gamma * lmbda
                        else:
                            _s.E[_a] = 0

                s = s_new
                a = a_new
                steps_left -= 1


if __name__ == '__main__':
    main()
