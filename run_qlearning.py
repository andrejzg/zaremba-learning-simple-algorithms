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
    alpha = 0.2
    gamma = 1

    for epoch in range(epochs):
        complexity = (epoch + 1) * 4 + 2
        acc = 0
        step = 1

        model.partial_fit(data.char2vec['r'], np.array([-1]), classes=data.classes)
        print('\n\n======= EPOCH\t{}/{}\tcomplexity:{} ======='.format(epoch + 1, epochs, complexity))

        while acc < 1.0:
            # new task
            input_tape = Grid(task=tasks.reverse, data=data, complexity=complexity)

            # initial state
            s = input_tape.start

            # initial action
            a = 'r'
            a_vec = data.char2vec[a]

            # steps left to complete task
            steps_left = complexity

            # print('\n\ncomplexity: {} solution: {}'.format(complexity, input_tape.outputs.chars))

            while True:
                # read input tape
                i = s.vec

                # combine input + prev action into input x
                x = (a_vec + i).reshape(1, -1)

                # expected output y
                y = input_tape.outputs.vecs.pop(0)

                # model output y_pred
                y_pred = model.predict(x)

                # fit model using one SGD step
                model.partial_fit(x, y, classes=data.classes)

                if y != -1 and y == y_pred:
                    print('HOORAH!')

                # generate reward signal
                if y == y_pred:
                    reward = 1
                    # print('CORRECT! {}\t\ty: {}\ty_pred: {}'.format(a, y[0], y_pred[0]))
                else:
                    # print('WRONG! {}\t\ty: {}\ty_pred: {}'.format(a, y[0], y_pred[0]))
                    break  # if reward is 0 break learning

                # pick an action to take
                a = policy.e_greedy(s, 0.2)
                a_vec = data.char2vec[a]  # the action's vector representation

                # take the action
                s_new = s.take_action(a)

                s.Q[a] -= alpha * (
                    s.Q[a] - (reward / steps_left + gamma * s_new.Q[policy.greedy(s_new)] / steps_left))

                s = s_new
                step += 1
                steps_left -= 1


if __name__ == '__main__':
    main()
