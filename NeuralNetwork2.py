import numpy
from numpy import array, dot, exp, random
from heapq import nsmallest
import os
import audio

if __name__ == "__main__":

    training_inputs, training_outputs = audio.LoadAudioTrainingData()
    #training_inputs = numpy.squeeze(training_inputs)

    number_layers = 2

    #layer1_inputs = 3
    layer1_inputs = 12
    #layer1_neurons = 4 #our hidden layer
    layer1_neurons = 7  # our hidden layer

    #layer2_inputs = 4
    layer2_inputs = 7
    layer2_neurons = 5 #this is our predicted output


    #seed the random number generator and create a 3x4 matrix of random numbers and a 4x1 matrix between [-1, 1]
    #random.random produces number between [0,1] so multiple 2 and substract 1 keeps numbers between [-1,1]
    #example: random = .25, 2 * .25 = .5, .5 - 1 = -0.5
    numpy.random.seed(1)

    layer1_synaptic_weights = 2 * numpy.random.random((layer1_inputs,layer1_neurons)) - 1
    print("randomly selected starting layer 1 synaptic weights")
    print(layer1_synaptic_weights.shape)
    print(layer1_synaptic_weights)
    print('\n')

    layer2_synaptic_weights = 2 * numpy.random.random((layer2_inputs, layer2_neurons)) - 1
    print("randomly selected starting layer 2 synaptic weights")
    print(layer2_synaptic_weights.shape)
    print(layer2_synaptic_weights)
    print('\n')

    # this is our ground truth data.  Each row on inputs is an input vector
    # and the Transposed output is 1 output for each input row
    #training_inputs = array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]])
    print("my training data input")
    print(training_inputs.shape)
    print(training_inputs)
    print('\n')

    #training_outputs = array([[0, 1, 1, 1, 1, 0, 0]]).T
    #training_outputs = training_outputs.T
    #training_outputs = numpy.expand_dims(training_outputs, axis=1)
    print("my training data output")
    print(training_outputs.shape)
    print(training_outputs)
    print('\n')

    number_of_training_iterations = 100000

    # /////////////
    # time to train
    # //////////////
    for iteration in range(number_of_training_iterations):
        print(iteration)

        # calculate weigted sum of each input vector = training_inputs[i] * synaptic weights
        #do this for both layers where layer 1 feeds layer 2
        layer1_sum_weighted_inputs = dot(training_inputs, layer1_synaptic_weights)
        print("this is the sum of my layer 1 weighted inputs")
        print(layer1_sum_weighted_inputs)
        print('\n')

        #since layer 2 takes actual output of layer 1 we need to do sigmoid first
        # normalize between [0, 1] with a sigmoid function
        layer1_sigmoid_normalized_output = 1 / (1 + exp(-layer1_sum_weighted_inputs))
        print("my output is now the sum of layer 1 weighted inputs normalized by a sigmoid function")
        print(layer1_sigmoid_normalized_output)
        print('\n')

        #now feed layer 1 output into layer 2
        layer2_sum_weighted_inputs = dot(layer1_sigmoid_normalized_output, layer2_synaptic_weights)
        print("this is the sum of my layer 2 weighted inputs")
        print(layer2_sum_weighted_inputs)
        print('\n')

        #now we need to sigmoid normalization for layer 2
        #THIS IS OUR OUTPUT PREDICTION
        layer2_sigmoid_normalized_output = 1 / (1 + exp(-layer2_sum_weighted_inputs))
        print("my output is now the sum of layer 2 weighted inputs normalized by a sigmoid function")
        print(layer2_sigmoid_normalized_output)
        print('\n')

        # //////////////////////////
        # time to adjust our weights
        # //////////////////////////

        # Calculate the error for layer 2 (The difference between the desired output
        # and the predicted output).
        layer2_error = training_outputs - layer2_sigmoid_normalized_output
        print("this is the layer 2 error: predicted output and actual output")
        print(layer2_error)
        print('\n')

        # Calculate the sigmoid derivative of layer 2 output i.e. predicted output
        layer2_sigmoid_derivative = layer2_sigmoid_normalized_output * (1 - layer2_sigmoid_normalized_output)
        print("this is the derivative of the layer 2 predicted output")
        print(layer2_sigmoid_derivative)
        print('\n')

        # pre-multiply error with sigmoid_derivative before multiplying wih inputs
        layer2_error_times_sigmoid = layer2_error * layer2_sigmoid_derivative
        print("this is the pre-multiplied layer 2 error and sigmoid_derivative")
        print(layer2_error_times_sigmoid)
        print('\n')

        # Calculate the error for layer 1 (By looking at the weights in layer 1,
        # we can determine by how much layer 1 contributed to the error in layer 2).
        layer1_error = dot(layer2_error_times_sigmoid,layer2_synaptic_weights.T )
        print("this is the layer 1 error: predicted output and actual output")
        print(layer1_error)
        print('\n')

        # Calculate the sigmoid derivative of layer 2 output i.e. predicted output
        layer1_sigmoid_derivative = layer1_sigmoid_normalized_output * (1 - layer1_sigmoid_normalized_output)
        print("this is the derivative of the layer 1 predicted output")
        print(layer1_sigmoid_derivative)
        print('\n')

        layer1_error_times_sigmoid = layer1_error * layer1_sigmoid_derivative
        print("this is the pre-multiplied layer 1 error and sigmoid_derivative")
        print(layer1_error_times_sigmoid)

        # time to figure out our final adjustments
        layer1_adjustments = dot(training_inputs.T, layer1_error_times_sigmoid)
        print("figure out our layer 1 adjustments by multiple each input times errror and sigmoud")
        print(layer1_adjustments)
        print('\n')

        layer2_adjustments = dot(layer1_sigmoid_normalized_output.T, layer2_error_times_sigmoid)
        print("figure out our layer 2 adjustments by multiple each input times errror and sigmoud")
        print(layer2_adjustments)
        print('\n')

        # calculate updated weights
        layer1_synaptic_weights += layer1_adjustments
        print("adjust layer 1 weights before iterating again")
        print(layer1_synaptic_weights)
        print('\n')

        layer2_synaptic_weights += layer2_adjustments
        print("adjust layer 2 weights before iterating again")
        print(layer2_synaptic_weights)
        print('\n')

        os.system('cls')

    # Test the neural network with a new situation.
    print("Let's Considering a new situation [A: ")
    test_data = audio.LoadAudioTestData('a_test.wav')
    test_data = numpy.squeeze(test_data)
    test_data = numpy.expand_dims(test_data, axis=1 )
    test_data = test_data.T
    print(test_data.shape)
    print(test_data)

    # calculate weigted sum of each input vector = training_inputs[i] * synaptic weights
    # do this for both layers where layer 1 feeds layer 2
    #layer1_sum_weighted_inputs = dot(array([1, 1, 0]), layer1_synaptic_weights)
    layer1_sum_weighted_inputs = dot(test_data, layer1_synaptic_weights)
    print((layer1_synaptic_weights))

    # since layer 2 takes actual output of layer 1 we need to do sigmoid first
    # normalize between [0, 1] with a sigmoid function
    layer1_sigmoid_normalized_output = 1 / (1 + exp(-layer1_sum_weighted_inputs))
    print(layer1_sigmoid_normalized_output)

    # now feed layer 1 output into layer 2
    layer2_sum_weighted_inputs = dot(layer1_sigmoid_normalized_output, layer2_synaptic_weights)
    print(layer2_sum_weighted_inputs)

    # now we need to sigmoid normalization for layer 2
    # THIS IS OUR OUTPUT PREDICTION
    layer2_sigmoid_normalized_output = 1 / (1 + exp(-layer2_sum_weighted_inputs))
    print("my output prediction")
    print(layer2_sigmoid_normalized_output)
    print('\n')

    print(layer2_sigmoid_normalized_output.argmax())
    print(audio.letters(layer2_sigmoid_normalized_output.argmax()))

    # Test the neural network with a new situation.
    print("Let's Considering a new situation [E: ")
    test_data = audio.LoadAudioTestData('e_test.wav')
    test_data = numpy.squeeze(test_data)
    test_data = numpy.expand_dims(test_data, axis=1 )
    test_data = test_data.T
    print(test_data.shape)
    print(test_data)

    # calculate weigted sum of each input vector = training_inputs[i] * synaptic weights
    # do this for both layers where layer 1 feeds layer 2
    #layer1_sum_weighted_inputs = dot(array([1, 1, 0]), layer1_synaptic_weights)
    layer1_sum_weighted_inputs = dot(test_data, layer1_synaptic_weights)
    print((layer1_synaptic_weights))

    # since layer 2 takes actual output of layer 1 we need to do sigmoid first
    # normalize between [0, 1] with a sigmoid function
    layer1_sigmoid_normalized_output = 1 / (1 + exp(-layer1_sum_weighted_inputs))
    print(layer1_sigmoid_normalized_output)

    # now feed layer 1 output into layer 2
    layer2_sum_weighted_inputs = dot(layer1_sigmoid_normalized_output, layer2_synaptic_weights)
    print(layer2_sum_weighted_inputs)

    # now we need to sigmoid normalization for layer 2
    # THIS IS OUR OUTPUT PREDICTION
    layer2_sigmoid_normalized_output = 1 / (1 + exp(-layer2_sum_weighted_inputs))
    print("my output prediction")
    print(layer2_sigmoid_normalized_output)
    print('\n')

    print(layer2_sigmoid_normalized_output.argmax())
    print(audio.letters(layer2_sigmoid_normalized_output.argmax()))

    # Test the neural network with a new situation.
    print("Let's Considering a new situation [I: ")
    test_data = audio.LoadAudioTestData('i_test.wav')
    test_data = numpy.squeeze(test_data)
    test_data = numpy.expand_dims(test_data, axis=1 )
    test_data = test_data.T
    print(test_data.shape)
    print(test_data)

    # calculate weigted sum of each input vector = training_inputs[i] * synaptic weights
    # do this for both layers where layer 1 feeds layer 2
    #layer1_sum_weighted_inputs = dot(array([1, 1, 0]), layer1_synaptic_weights)
    layer1_sum_weighted_inputs = dot(test_data, layer1_synaptic_weights)
    print((layer1_synaptic_weights))

    # since layer 2 takes actual output of layer 1 we need to do sigmoid first
    # normalize between [0, 1] with a sigmoid function
    layer1_sigmoid_normalized_output = 1 / (1 + exp(-layer1_sum_weighted_inputs))
    print(layer1_sigmoid_normalized_output)

    # now feed layer 1 output into layer 2
    layer2_sum_weighted_inputs = dot(layer1_sigmoid_normalized_output, layer2_synaptic_weights)
    print(layer2_sum_weighted_inputs)

    # now we need to sigmoid normalization for layer 2
    # THIS IS OUR OUTPUT PREDICTION
    layer2_sigmoid_normalized_output = 1 / (1 + exp(-layer2_sum_weighted_inputs))
    print("my output prediction")
    print(layer2_sigmoid_normalized_output)
    print('\n')

    print(layer2_sigmoid_normalized_output.argmax())
    print(audio.letters(layer2_sigmoid_normalized_output.argmax()))

    # Test the neural network with a new situation.
    print("Let's Considering a new situation [O: ")
    test_data = audio.LoadAudioTestData('o_test.wav')
    test_data = numpy.squeeze(test_data)
    test_data = numpy.expand_dims(test_data, axis=1 )
    test_data = test_data.T
    print(test_data.shape)
    print(test_data)

    # calculate weigted sum of each input vector = training_inputs[i] * synaptic weights
    # do this for both layers where layer 1 feeds layer 2
    #layer1_sum_weighted_inputs = dot(array([1, 1, 0]), layer1_synaptic_weights)
    layer1_sum_weighted_inputs = dot(test_data, layer1_synaptic_weights)
    print((layer1_synaptic_weights))

    # since layer 2 takes actual output of layer 1 we need to do sigmoid first
    # normalize between [0, 1] with a sigmoid function
    layer1_sigmoid_normalized_output = 1 / (1 + exp(-layer1_sum_weighted_inputs))
    print(layer1_sigmoid_normalized_output)

    # now feed layer 1 output into layer 2
    layer2_sum_weighted_inputs = dot(layer1_sigmoid_normalized_output, layer2_synaptic_weights)
    print(layer2_sum_weighted_inputs)

    # now we need to sigmoid normalization for layer 2
    # THIS IS OUR OUTPUT PREDICTION
    layer2_sigmoid_normalized_output = 1 / (1 + exp(-layer2_sum_weighted_inputs))
    print("my output prediction")
    print(layer2_sigmoid_normalized_output)
    print('\n')

    print(layer2_sigmoid_normalized_output.argmax())
    print(audio.letters(layer2_sigmoid_normalized_output.argmax()))

    # Test the neural network with a new situation.
    print("Let's Considering a new situation [U: ")
    test_data = audio.LoadAudioTestData('u_test.wav')
    test_data = numpy.squeeze(test_data)
    test_data = numpy.expand_dims(test_data, axis=1 )
    test_data = test_data.T
    print(test_data.shape)
    print(test_data)

    # calculate weigted sum of each input vector = training_inputs[i] * synaptic weights
    # do this for both layers where layer 1 feeds layer 2
    #layer1_sum_weighted_inputs = dot(array([1, 1, 0]), layer1_synaptic_weights)
    layer1_sum_weighted_inputs = dot(test_data, layer1_synaptic_weights)
    print((layer1_synaptic_weights))

    # since layer 2 takes actual output of layer 1 we need to do sigmoid first
    # normalize between [0, 1] with a sigmoid function
    layer1_sigmoid_normalized_output = 1 / (1 + exp(-layer1_sum_weighted_inputs))
    print(layer1_sigmoid_normalized_output)

    # now feed layer 1 output into layer 2
    layer2_sum_weighted_inputs = dot(layer1_sigmoid_normalized_output, layer2_synaptic_weights)
    print(layer2_sum_weighted_inputs)

    # now we need to sigmoid normalization for layer 2
    # THIS IS OUR OUTPUT PREDICTION
    layer2_sigmoid_normalized_output = 1 / (1 + exp(-layer2_sum_weighted_inputs))
    print("my output prediction")
    print(layer2_sigmoid_normalized_output)
    print('\n')

    print(layer2_sigmoid_normalized_output.argmax())
    print(audio.letters(layer2_sigmoid_normalized_output.argmax()))

    #true_output = nsmallest(1, training_outputs, key=lambda x: abs(x - layer2_sigmoid_normalized_output))
    #print(true_output[0])
    #print(audio.VOWELS(true_output[0]).name)

