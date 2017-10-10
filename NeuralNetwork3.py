import numpy
from numpy import array, dot, exp, random
import math
import audio

numpy.seterr(all='ignore')

class Layer():
    def __init__(self, inputs, neurons, activation, activation_derivative):
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

class NeuralNetwork():
    def __init__(self, layer1, layer2, layer3, learning_rate, learning_rate_decay, momentum):
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.momentum = momentum

        self.l1_inputs = numpy.ones((self.layer1.inputs, self.layer1.neurons))
        self.l2_hidden = numpy.ones((self.layer2.inputs, self.layer2.neurons))
        self.l3_output = numpy.ones((self.layer3.inputs, self.layer3.neurons))

        self.l2_hidden_adjustment = numpy.zeros((self.layer2.inputs, self.layer2.neurons))
        self.l3_output_adjustment = numpy.zeros((self.layer3.inputs, self.layer3.neurons))

    def Sigmoid_Activation(self,X):
        X = numpy.nan_to_num(X)
        return 1 / (1 + numpy.exp(-X))

    def Sigmoid_Activation_Derivative(self, X):
        X = numpy.nan_to_num(X)
        return X * (1.0 - X)

    def Tanh_Activation(self, X):
        X = numpy.nan_to_num(X)
        return numpy.tanh(X)

    def Tanh_Activation_Deriv(self, X):
        X = numpy.nan_to_num(X)
        return 1 - X * X
        #return 1.0 - math.tanh(X) ** 2

    def ReLu_Activation(self, X):
        X = numpy.nan_to_num(X)
        return numpy.maximum(X, 0, X)
        #for x in numpy.nditer(X, op_flags=['readwrite']):
        #    if x < 0: # or x != numpy.NaN:
        #        x[...] = 0.0
        #    else:
        #        x[...] = x
        #return X

    def ReLu_Activation_Derivative(self, X):
        X = numpy.nan_to_num(X)
        for x in numpy.nditer(X, op_flags=['readwrite']):
            if x < 0: # or x != numpy.NaN:
                x[...] = 0.0
            else:
                x[...] = 1.0
        return X

    def Leaky_ReLu_Activation(self, X):
        X = numpy.nan_to_num(X)
        return numpy.maximum(0.1 * X, X)

        #for x in numpy.nditer(X, op_flags=['readwrite']):
        #    if x < 0 or x != numpy.NaN:
        #        x[...] = 0.01 * x
        #    else:
        #        x[...] = x
        #return X

    def Leaky_ReLu_Activation_Derivative(self, X):
        X = numpy.nan_to_num(X)
        gradients = 1. * (X > 0)
        gradients[gradients == 0] = .1
        return gradients

        #for x in numpy.nditer(X, op_flags=['readwrite']):
        #    if x < 0 or x != numpy.NaN:
        #        x[...] = 0.01
        #    else:
        #        x[...] = 1.0
        #return X

    def softmax(self, X):
        e = numpy.exp(X - numpy.amax(X))
        dist = e / numpy.sum(e)
        return dist

    def softmax_derivative(self, X):
        return

    def Test(self, inputs, outputs):
        for p, q in zip(inputs, outputs):
            o = self.Feed_Forward(p)
            print('ACTUAL:', numpy.array([q]).argmax(), '->', 'PREDICTED:', numpy.array([o]).argmax())

    def Train(self, inputs, outputs, iterations):
        for iteration in range(iterations):
            error = 0.0

            # random.shuffle(patterns)
            # turn off random
            randomize = numpy.arange(len(inputs))
            numpy.random.shuffle(randomize)
            inputs = inputs[randomize]
            outputs = outputs[randomize]

            self.Feed_Forward(inputs)
            error = self.Back_Propogate(outputs)
            error = numpy.average(error)
            if iteration % 10 == 0:
                print('error %-.5f' % error)
            # learning rate decay
            self.learning_rate = self.learning_rate * (
            self.learning_rate / (self.learning_rate + (self.learning_rate * self.learning_rate_decay)))

    def Back_Propogate(self, outputs):

        output_deltas = numpy.zeros((self.layer1.inputs, self.layer3.neurons))
        l3_output_error = -(outputs - self.l3_output)
        if self.layer3.activation_derivative == NeuralNetwork.Sigmoid_Activation_Derivative:
            output_deltas = self.layer3.activation_derivative(self, self.l3_output) * l3_output_error
        elif self.layer3.activation_derivative == NeuralNetwork.softmax_derivative:
            output_deltas = l3_output_error

        hidden_deltas = numpy.zeros((self.layer1.inputs, self.layer2.neurons))
        l2_hidden_error = output_deltas.dot(self.layer3.synaptic_weights.T)
        hidden_deltas = self.layer2.activation_derivative(self, self.l2_hidden) * l2_hidden_error

        adjustment1 = self.l2_hidden.T.dot(output_deltas)
        self.layer3.synaptic_weights = self.layer3.synaptic_weights - (adjustment1 * self.learning_rate) #+ self.l3_output_adjustment * self.momentum
        self.l3_output_adjustment = adjustment1

        adjustment2 = self.l1_inputs.T.dot(hidden_deltas)
        self.layer2.synaptic_weights = self.layer2.synaptic_weights - (adjustment2 * self.learning_rate) #+ self.l2_hidden_adjustment * self.momentum
        self.l2_hidden_adjustment = adjustment2

        # calculate error
        error = 0.0
        error = numpy.sum(0.5 * (outputs - self.l3_output) ** 2, axis=0)
        return error


    def Feed_Forward(self, inputs):
        self.l1_inputs[:,0:12] = inputs
        self.l2_hidden = self.layer2.activation(self, dot(self.l1_inputs, self.layer2.synaptic_weights))
        self.l3_output = self.layer3.activation(self, dot(self.l2_hidden, self.layer3.synaptic_weights))
        return  self.l3_output


if __name__ == "__main__":

    training_inputs, training_outputs, validation_inputs, validation_outputs = audio.LoadAudioTrainingDataFromFile('file_list.csv', 25, 13)

    #need to null activation layers
    input_layer = Layer(inputs=training_inputs.shape[0], neurons=training_inputs.shape[1] + 1, activation=NeuralNetwork.Tanh_Activation, activation_derivative=NeuralNetwork.Tanh_Activation_Deriv)

    hidden_layer = Layer(inputs=training_inputs.shape[1] + 1, neurons=200, activation=NeuralNetwork.Tanh_Activation, activation_derivative=NeuralNetwork.Tanh_Activation_Deriv)
    hidden_layer.Initialize_Synaptic_Weights()

    output_layer = Layer(inputs=200, neurons=training_outputs.shape[1], activation=NeuralNetwork.Sigmoid_Activation, activation_derivative=NeuralNetwork.Sigmoid_Activation_Derivative)
    output_layer.Initialize_Synaptic_Weights()

    nnet = NeuralNetwork(layer1=input_layer, layer2=hidden_layer, layer3=output_layer, learning_rate=0.001, learning_rate_decay=0.0001, momentum=0.5)

    nnet.Train(training_inputs, training_outputs, 25000)

    print("TRAINING VALIDATION:")
    nnet.Test(training_inputs, training_outputs)

    print("TEST VALIDATION:")
    nnet.Test(validation_inputs, validation_outputs)
