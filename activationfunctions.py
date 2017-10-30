import numpy
from numpy import array, dot, exp, random

def Sigmoid_Activation(X):
    X = numpy.nan_to_num(X)
    return 1 / (1 + numpy.exp(-X))


def Sigmoid_Activation_Derivative(X):
    X = numpy.nan_to_num(X)
    return X * (1.0 - X)

def Swish_Activation(X):
    Y = X * Sigmoid_Activation(X)
    Y = numpy.nan_to_num(Y)
    return Y

def Swish_Activation_Derivative(X):
    Y = Swish_Activation(X) + Sigmoid_Activation(X) * (1 - Swish_Activation(X))
    Y = numpy.nan_to_num(Y)
    return Y

def Tanh_Activation(X):
    X = numpy.nan_to_num(X)
    return numpy.tanh(X)


def Tanh_Activation_Deriv(X):
    X = numpy.nan_to_num(X)
    return 1 - X * X
    # return 1.0 - math.tanh(X) ** 2


def ReLu_Activation(X):
    X = numpy.nan_to_num(X)
    return numpy.maximum(X, 0, X)
    # for x in numpy.nditer(X, op_flags=['readwrite']):
    #    if x < 0: # or x != numpy.NaN:
    #        x[...] = 0.0
    #    else:
    #        x[...] = x
    # return X


def ReLu_Activation_Derivative(X):
    X = numpy.nan_to_num(X)
    for x in numpy.nditer(X, op_flags=['readwrite']):
        if x < 0:  # or x != numpy.NaN:
            x[...] = 0.0
        else:
            x[...] = 1.0
    return X


def Leaky_ReLu_Activation(X):
    #X = numpy.nan_to_num(X)
    return numpy.maximum(0.1 * X, X)

    # for x in numpy.nditer(X, op_flags=['readwrite']):
    #    if x < 0 or x != numpy.NaN:
    #        x[...] = 0.01 * x
    #    else:
    #        x[...] = x
    # return X


def Leaky_ReLu_Activation_Derivative(X):
    #X = numpy.nan_to_num(X)
    gradients = 1. * (X > 0)
    gradients[gradients == 0] = .1
    return gradients

    # for x in numpy.nditer(X, op_flags=['readwrite']):
    #    if x < 0 or x != numpy.NaN:
    #        x[...] = 0.01
    #    else:
    #        x[...] = 1.0
    # return X


def softmax(X):
    e = numpy.exp(X - numpy.amax(X))
    dist = e / numpy.sum(e) #, axis=1, keepdims=True)
    return dist



def softmax_derivative(X):
    return

def Oland_Et_Al(X):
    alpha = 0.01
    Beta = 0.4
    return alpha * numpy.power(X, 3) + Beta

def Oland_Et_Al_Derivative(X):
    alpha = 0.01
    return alpha * numpy.exp(X)
