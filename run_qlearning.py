import numpy as np
import world.policy as policy

from sklearn import neural_network

from world.grid import Grid
from world import data
from world import tasks


def main():
    ########################################################################
    #                                   TRAIN
    ########################################################################

    data = data.Data()
    epochs = 10
    model = neural_network.MLPClassifier(activation='logistic', hidden_layer_sizes=(200,))
    alpha = 0.2
    gamma = 1

    for epoch in range(epochs):
        complexity = (epoch + 1) * 4 + 2
        acc = 0
        epoch_iter = 0

        model.partial_fit(data.char2vec['r'], np.array([-1]), classes=data.classes)
        # print('\n\n======= EPOCH\t{}/{}\tcomplexity:{} ======='.format(epoch + 1, epochs, complexity))

        input_tape = Grid(1, complexity)

        while acc < 1.0:
            epoch_iter += 1

            # generate data
            inputs, actions, outputs, in_tape, solution, _ = data.task(tasks.reverse, complexity)
            input_tape.populate(inputs)
            input_tape.solution = solution
            input_tape.reset_head()

            # initial state
            s = input_tape.start

            # print('\n\ncomplexity: {} solution: {}'.format(complexity, solution))

            while True:
                # which action to take?
                a = policy.e_greedy(s, 0.2)

                # take action
                s_new = s.take_action(a)

                # action vector
                a_vec = data.char2vec[a]

                # read input tape
                i = s.data

                # combine input + action vector into input x
                x = (a_vec + i).reshape(1, -1)

                # expected output y
                y = outputs.pop(0)
                solution.pop(0)

                # get model prediction
                y_pred = model.predict(x)

                # partial fit model using a single SGD step
                model.partial_fit(x, y, classes=data.classes)

                if y != -1 and y == y_pred:
                    print('HOORAH!')

                # generate reward signal
                if y == y_pred:
                    reward = 1
                    # print('CORRECT! {}\t\ty: {}\ty_pred: {}'.format(a, y[0], y_pred[0]))
                else:
                    reward = 0
                    # print('WRONG! {}\t\ty: {}\ty_pred: {}'.format(a, y[0], y_pred[0]))

                # if reward is 0 terminate
                if reward == 0:
                    break

                s.Q[a] -= alpha * (
                    s.Q[a] - (reward / len(solution) + gamma * s_new.Q[policy.greedy(s_new)] / len(solution)))

                s = s_new

            # print(in_tape)

            # Test model every now and again
            if epoch_iter % 100 == 0:
                acc = test(complexity, data, model, 100)
                print(acc)

    ########################################################################
    #                                   TEST
    ########################################################################

    print('\n\nRUNNING TESTS...')
    for i in range(10):
        complexity = (i + 1) * 8 + 2
        test(complexity, data, model, 1, verbose=True)


# Helper functions

def test(complexity, data, model, runs, verbose=False):
    # 100 hold-out examples of current length test

    if verbose:
        print('\n\ncomplexity: {}'.format(complexity))

    accuracies = []
    for _ in range(runs):
        inputs, actions, outputs, _, solution, _ = data.task(tasks.reverse, complexity)

        input_tape = Grid(1, complexity)
        input_tape.populate(inputs)
        input_tape.solution = solution

        predictions = []

        # initial action
        a = np.zeros(inputs[0].shape[0])

        while True:
            # read input
            i = input_tape.head.data

            # combine into input
            x = (a + i).reshape(1, -1)
            y = model.predict(x)
            predictions.append(y)

            if input_tape.step() is None:
                break

            # new prev action
            a = actions.pop(0)

        accuracies.append(similarity(predictions, outputs))

        if verbose:
            print('\nexpected:\t{}'.format([o.tolist()[0] for o in outputs]))
            print('model:\t\t{}'.format([p.tolist()[0] for p in predictions]))

    return np.mean(accuracies)


def similarity(list1, list2):
    assert len(list1) == len(list2)
    match = 0
    for a, b in zip(list1, list2):
        if a == b:
            match += 1
    return match / len(list1)


if __name__ == '__main__':
    main()
