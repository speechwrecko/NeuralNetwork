import sys
from librosa import load, util, feature
from librosa.core import stft, magphase
import numpy
from numpy import array, dot, exp, random
from enum import Enum
from sklearn import preprocessing
import csv
import inspect
import os

class numbers(Enum):
    zero = 0
    one = 1
    two = 2
    three = 3
    four = 4
    five = 5
    six = 6
    seven = 7
    eight = 8
    nine = 9

class digits:
    zero    = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    one     = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    two     = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    three   = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    four    = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    five    = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    six     = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    seven   = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    eight   = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    nine    = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]


def LoadAudioTrainingDataFromFile(file, validation_size, nmfcc):

    #initiliaze arrays
    mfcc_input = []
    mag_input = []
    digit = []
    output = []

    #read in CSV file with file names
    #????? Should probably require output as a seperate column instead of reading from file name
    with open(file, 'r') as f:
        reader = csv.reader(f)
        file_list = list(reader)

    #load in audio and file name data
    dirname = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
    for files in file_list:
        relative_path = 'recordings/' + files[0]
        file_name = os.path.join(os.path.dirname(__file__), relative_path)
        y, sr = load(file_name)
        filesize = sys.getsizeof(y)

        spectrum = stft(y, hop_length=int(filesize / 2))
        mag, phase = magphase(spectrum)
        mag_input.append(mag)

        mfcc = feature.mfcc(y, sr, n_mfcc=nmfcc, hop_length=int(filesize / 2))
        mfcc = mfcc[1:nmfcc]
        mfcc_input.append(mfcc)

        digit.append(files[0][0])

    #build array of one hot vectors for output based on 1st character in file name
    for num in digit:
        if num == '0':
            output.append(digits.zero)
        if num == '1':
            output.append(digits.one)
        if num == '2':
            output.append(digits.two)
        if num == '3':
            output.append(digits.three)
        if num == '4':
            output.append(digits.four)
        if num == '5':
            output.append(digits.five)
        if num == '6':
            output.append(digits.six)
        if num == '7':
            output.append(digits.seven)
        if num == '8':
            output.append(digits.eight)
        if num == '9':
            output.append(digits.nine)

    #normalize data to between 0 and 1
    training_input = numpy.asarray(mfcc_input, dtype=numpy.float64)
    training_input = numpy.squeeze(training_input)
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    training_input = min_max_scaler.fit_transform(training_input)

    training_output = numpy.asarray(output, dtype=numpy.float64)
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    training_output = min_max_scaler.fit_transform(training_output)

    #randommize before dividing test/train sets:
    randomize = numpy.arange(len(training_input))
    numpy.random.shuffle(randomize)
    training_input = training_input[randomize]
    training_output = training_output[randomize]

    #pull out validation set
    validation_input = training_input[0:validation_size,:]
    validation_output = training_output[0:validation_size,:]

    return training_input, training_output, validation_input,validation_output







