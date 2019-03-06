__author__ = 'raza'
import numpy as np


def softmax(_input):
    """
    Applies softmax function and computes the output
    """
    score = np.exp(_input)
    _sum = np.sum(score)

    return np.divide(score, _sum)


def tanh_derivative(_input):
    """
    Computes and returns derivative of a tangent hyperbolic function
    1 - (f'(x))^2
    """
    return 1. - np.square(_input)


def concat_with_bias(left_vector, right_vector):
    """
    Stacks and returns the two given vectors vertically.
    Adds an always-on input for the bias units.
    """
    return np.vstack((left_vector, right_vector, 1.))


def concat(left_vector, right_vector):
    """
    Stacks and returns the two given vectors vertically
    """
    return np.vstack((left_vector, right_vector))


def init_random(mean, var, shape):
    return np.random.normal(mean, var, size=shape)