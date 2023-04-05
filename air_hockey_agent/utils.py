import numpy as np


def to_vector_coordinate(r, theta):
    """
    Convert polar coordinate to vector coordinate
    :param r: radius
    :param theta: angle
    """
    return r * np.cos(theta), r * np.sin(theta)


def to_polar_coordinate(x, y):
    """
    Convert vector coordinate to polar coordinate
    :param x: x coordinate
    :param y: y coordinate
    """
    return np.sqrt((x ** 2 + y ** 2)), np.arctan2(y / x)
