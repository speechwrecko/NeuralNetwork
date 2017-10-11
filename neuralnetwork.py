import numpy
from numpy import array, dot, exp, random
import layers
import activationfunctions
import audio
import inspect
import os

numpy.seterr(all='ignore')

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
        if self.layer3.activation_derivative == activationfunctions.Sigmoid_Activation_Derivative:
            output_deltas = self.layer3.activation_derivative(self.l3_output) * l3_output_error
        elif self.layer3.activation_derivative == activationfunctions.softmax_derivative:
            output_deltas = l3_output_error

        hidden_deltas = numpy.zeros((self.layer1.inputs, self.layer2.neurons))
        l2_hidden_error = output_deltas.dot(self.layer3.synaptic_weights.T)
        hidden_deltas = self.layer2.activation_derivative(self.l2_hidden) * l2_hidden_error

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
        self.l1_inputs[:,0:self.layer1.neurons-1] = inputs
        self.l2_hidden = self.layer2.activation(dot(self.l1_inputs, self.layer2.synaptic_weights))
        self.l3_output = self.layer3.activation(dot(self.l2_hidden, self.layer3.synaptic_weights))
        return  self.l3_output


if __name__ == "__main__":

    dirname = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
    file_name = os.path.join(os.path.dirname(__file__), './recordings/file_list.csv')

    training_inputs, training_outputs, validation_inputs, validation_outputs = audio.LoadAudioTrainingDataFromFile(csv_file_name=file_name, validation_size=25, nmfcc=13)

    #need to null activation layers
    input_layer = layers.Layer(inputs=training_inputs.shape[0], neurons=training_inputs.shape[1] + 1)

    hidden_layer = layers.Layer(inputs=training_inputs.shape[1] + 1, neurons=200, activation=activationfunctions.Tanh_Activation, activation_derivative=activationfunctions.Tanh_Activation_Deriv)
    hidden_layer.Initialize_Synaptic_Weights()

    output_layer = layers.Layer(inputs=200, neurons=training_outputs.shape[1], activation=activationfunctions.Sigmoid_Activation, activation_derivative=activationfunctions.Sigmoid_Activation_Derivative)
    output_layer.Initialize_Synaptic_Weights()

    nnet = NeuralNetwork(layer1=input_layer, layer2=hidden_layer, layer3=output_layer, learning_rate=0.001, learning_rate_decay=0.0001, momentum=0.5)

    nnet.Train(training_inputs, training_outputs, 25000)

    print("TRAINING VALIDATION:")
    nnet.Test(training_inputs, training_outputs)

    print("TEST VALIDATION:")
    nnet.Test(validation_inputs, validation_outputs)
