import numpy as np
from numpy import array, dot, exp, random
import audio


def nonlin(x, deriv=False):
    if (deriv == True):
        #return x * (1 - x)
        return 1 - X * X

    #return 1 / (1 + np.exp(-x))
    return np.tanh(X)


#X = np.array([[0, 0, 1],
#              [0, 1, 1],
#              [1, 0, 1],
#              [1, 1, 1]])

#y = np.array([[0],
#              [1],
#              [1],
#              [0]])

X, y = audio.LoadAudioTrainingDataFromFile('file_list.csv', 13)

np.random.seed(1)

# randomly initialize our weights with mean 0
syn0 = 2 * np.random.random((12, 100)) - 1
syn1 = 2 * np.random.random((100, 10)) - 1

#input_range = 1.0 / 12 ** (1 / 2)
#syn0 = np.random.normal(loc=0, scale=input_range, size=(12, 100))
#output_range = 1.0 / 10 ** (1 / 2)
#syn1 = np.random.normal(loc=0, scale=input_range, size=(100, 10))

for j in range(500):

    # Feed forward through layers 0, 1, and 2
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1))

    # how much did we miss the target value?
    l2_error = y - l2

    if (j % 10) == 0:
        print
        "Error:" + str(np.mean(np.abs(l2_error)))

    # in what direction is the target value?
    # were we really sure? if so, don't change too much.
    l2_delta = l2_error * nonlin(l2, deriv=True)

    # how much did each l1 value contribute to the l2 error (according to the weights)?
    l1_error = l2_delta.dot(syn1.T)

    # in what direction is the target l1?
    # were we really sure? if so, don't change too much.
    l1_delta = l1_error * nonlin(l1, deriv=True)

    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)
