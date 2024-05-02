"""
mnist_loader
~~~~~~~~~~~~

PARTS OF THIS FILE ORIGINATE FROM MICHAEL NIELSEN'S PROJECT FOUND HERE:
https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/mnist_loader.py

A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

#### Libraries
# Standard library
# import gzip
import pickle
from random import shuffle, uniform

# Third-party libraries
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray


def load_data() -> tuple:
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.

    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.

    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.

    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below."""
    
    with open("data/mnist.pkl", "rb") as f:
        training_data, validation_data, test_data = pickle.load(f, encoding="latin")

    return (training_data, validation_data, test_data)


def load_noise() -> tuple:
    """Load the uniform noise samples and split into groups of sizes equal
    to those of original datatset. 5000 training samples, 1000 validation
    samples and 1000 test samples as is the case for each digit."""
    
    with open("data/uniform_noise.pickle", "rb") as f:
        noise = pickle.load(f, encoding="latin")

    noise_training = noise[:5000]
    noise_validation = noise[5000:6000]
    noise_test = noise[6000:]

    return noise_training, noise_validation, noise_test


def load_data_wrapper(noise=False) -> tuple:
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.

    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.

    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.

    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""
    
    if noise:
        noise_training, noise_validation, noise_test = load_noise()
    
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (-1, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y, noise=noise) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    
    validation_inputs = [np.reshape(x, (-1, 1)) for x in va_d[0]]
    validation_results = [vectorized_result(y, noise=noise) for y in va_d[1]]
    validation_data = list(zip(validation_inputs, validation_results))
    
    test_inputs = [np.reshape(x, (-1, 1)) for x in te_d[0]]
    test_results = [vectorized_result(y, noise=noise) for y in te_d[1]]
    test_data = list(zip(test_inputs, test_results))
    
    if noise:
        validation_data += noise_validation
        training_data += noise_training
        test_data += noise_test
    
    return (training_data, validation_data, test_data)


def vectorized_result(j, noise=False):
    """Return a 10-dimensional unit vector with a 1.0 in the j-th
    position and zeroes elsewhere. This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    
    e = np.zeros((11 if noise else 10, 1))
    e[j] = 1.0
    return e






def uniform_image(size=28*28, image_only=False):
    random_input = [uniform(0, 1) for _ in range(size)]
    random_input = np.array(random_input).reshape(-1, 1)
    expected_output = vectorized_result(10, noise=True)
    
    if image_only:
        return random_input
    else:
        return random_input, expected_output



def get_digit(e):
    return np.argmax(e)

def get_first_10_digits(training_data, i=0):
    digits = []
    current_digit = 0

    while len(digits) < 10:
        if get_digit(training_data[i][1]) == current_digit:
            digits.append(training_data[i])
            current_digit += 1
        i = (i + 1) % len(training_data)

    return digits

def split_batches(data, size=100):
    shuffle(data)
    return [data[i : i + size] for i in range(0, len(data), size)]


def show_as_image(array: ndarray, save_image_as: str|None = None):
    plt.imshow(array.reshape((28, 28)), cmap="gray")
    plt.axis("off")
    plt.show()

    if save_image_as:
        plt.savefig(save_image_as)
