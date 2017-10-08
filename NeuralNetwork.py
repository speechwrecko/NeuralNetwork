import numpy
from numpy import array, dot, exp, random
import os

if __name__ == "__main__":

    #seed the random number generator and create a 3x1 matrix of random numbers between [-1, 1]
    #random.random produces number between [0,1] so multiple 2 and substract 1 keeps numbers between [-1,1]
    #example: random = .25, 2 * .25 = .5, .5 - 1 = -0.5
    numpy.random.seed(1)
    synaptic_weights = 2 * numpy.random.random((3,1)) - 1
    print("randomly selected starting synaptic weights")
    print(synaptic_weights)
    print('\n')

    #this is our ground truth data.  Each row on inputs is an input vector
    # and the Transposed output is 1 output for each input row
    training_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    print("my training data input")
    print(training_inputs)
    print('\n')

    training_outputs = array([[0, 1, 1, 0]]).T
    print("my training data output")
    print(training_outputs)
    print('\n')

    number_of_training_iterations = 10000

    # /////////////
    # time to train
    #//////////////
    for iteration in range(number_of_training_iterations):

        print(iteration)

        # calculate weigted sum of each input vector = training_inputs[i] * synaptic weights
        sum_weighted_inputs = dot(training_inputs, synaptic_weights)
        print("this is the sum of my weighted inputs")
        print(sum_weighted_inputs)
        print('\n')

        #normalize between [0, 1] with a sigmoid function
        sigmoid_normalized_output = 1 / (1 + exp(-sum_weighted_inputs))
        print("my output is now the sum of weighted inputs normalized by a sigmoid function")
        print(sigmoid_normalized_output)
        print('\n')

        #//////////////////////////
        #time to adjust our weights
        #//////////////////////////

        #calculate the error between predicted outputs and actual output
        error = training_outputs - sigmoid_normalized_output
        print("this is the error from my predicted output and actual output")
        print(error)
        print('\n')

        #Calculate the sigmoid derivative of predicted output
        sigmoid_derivative = sigmoid_normalized_output * (1 - sigmoid_normalized_output)
        print("this is the derivative of the predicted output")
        print(sigmoid_derivative)
        print('\n')

        #pre-multiply error with sigmoid_derivative before multiplying wih inputs
        error_times_sigmoid = error * sigmoid_derivative
        print("this is the pre-multiplied error and sigmoid_derivative")
        print(error_times_sigmoid)
        print('\n')

        #time to figure out our final adjustments
        adjustments = dot(training_inputs.T, error_times_sigmoid)
        print("figure out our adjustments by multiple each input times errror and sigmoud")
        print(adjustments)
        print('\n')

        #calculate updated weights
        synaptic_weights += adjustments
        print("adjust weights before iterating again")
        print(synaptic_weights)
        print('\n')

        os.system('cls')
