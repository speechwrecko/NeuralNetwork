import numpy
from numpy import array, dot, exp, random
import math

class Layer():
    def __init__(self, inputs, neurons, activation=None, activation_derivative=None):
        self.inputs = inputs
        self.neurons = neurons
        self.activation = activation
        self.activation_derivative = activation_derivative

        self.synaptic_weights = numpy.array([], dtype=numpy.float64)
        self.synaptic_weights = numpy.zeros((self.inputs, self.neurons))
        #test initialize to a constant
        #self.synaptic_weights = numpy.ones((self.inputs, self.neurons))

    def Initialize_Synaptic_Weights(self):
        # create randomized weights
        # use scheme from 'efficient backprop to initialize weights
        input_range = 1.0 / self.inputs ** (1 / 2)
        self.synaptic_weights = numpy.random.normal(loc=0, scale=input_range, size=(self.inputs, self.neurons))

    def Initialize_Synaptic_weights_Glorot(self):
        self.synaptic_weights= numpy.random.uniform(size=(self.inputs, self.neurons)) / numpy.sqrt(self.inputs)

    def Initialize_Synaptic_weights_Glorot_tanh(self):
            self.synaptic_weights = numpy.random.uniform(low=-(math.sqrt( 6 / (self.inputs + self.neurons))), high=math.sqrt( 6 / (self.inputs + self.neurons)), size=(self.inputs, self.neurons))

    def Initialize_Synaptic_weights_Glorot_sigmoid(self):
            self.synaptic_weights = numpy.random.uniform(low=-(4 * math.sqrt( 6 / (self.inputs + self.neurons))), high=4 * math.sqrt( 6 / (self.inputs + self.neurons)), size=(self.inputs, self.neurons))
