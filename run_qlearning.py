import numpy as np
from sklearn import neural_network

from world.grid import Grid
from world import datagen
from world import tasks


def main():
    ########################################################################
    #                                   TRAIN
    ########################################################################

    # TODO: implement Q-learning            // THU / FRI

    data = datagen.Data()
    classes = list('0123456789') + ['-1']

    acc = 0
    epochs = 20
    model = neural_network.MLPClassifier(activation='logistic', hidden_layer_sizes=(100,))

    for epoch in range(epochs):
        complexity = (epoch + 1) * 4 + 2
        acc = 0
        epoch_iter = 0

        print('\n\n======= EPOCH\t{}/{}\tcomplexity:{} ======='.format(epoch + 1, epochs, complexity))

        while acc < 1.0:
            epoch_iter += 1

            # generate data
            inputs, actions, outputs, solution = data.task(tasks.reverse, complexity)

            input_tape = Grid(1, complexity)
            input_tape.populate(inputs)
            input_tape.solution = solution

            # initial action
            a = np.zeros(inputs[0].shape[0])

            # keep taking actions until the end
            while True:
                # read input tape
                i = input_tape.head.data

                # combine input + action vectors into input x
                x = (a + i).reshape(1, -1)

                # expected output y
                y = outputs.pop(0)

                # partial fit model using a single SGD step
                model.partial_fit(x, y, classes=data.classes)

                # refactor into
                if input_tape.step() is None:
                    break

                # new prev action
                a = actions.pop(0)

            # Test model every now and again
            if epoch_iter % 1 == 0:
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
        inputs, actions, outputs, solution = data.task(tasks.reverse, complexity)

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
