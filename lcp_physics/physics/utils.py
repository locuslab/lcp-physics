import math
import os

import pygame

import torch
from torch.autograd import Variable


class Params:
    # Dimensions
    DIM = 2

    # Contact tolerance parameter
    DEFAULT_EPSILON = 0.1
    # Parallel contact tolerance
    DEFAULT_PAR_EPS = DEFAULT_EPSILON

    # Default simulation parameters
    DEFAULT_RESTITUTION = 0.3

    DEFAULT_FRIC_COEFF = 0.23
    DEFAULT_FRIC_DIRS = 2

    DEFAULT_FPS = 30
    DEFAULT_DT = 1.0 / DEFAULT_FPS

    DEFAULT_ENGINE = 'PdipmEngine'
    DEFAULT_COLLISION = 'DiffCollisionHandler'

    # Tensor type
    TENSOR_TYPE = torch.DoubleTensor

    # Post stabilization flag
    POST_STABILIZATION = False

    def __init__(self):
        pass


class Indices:
    X = 0
    Y = 1
    Z = 2

    def __init__(self):
        pass


class Recorder:
    def __init__(self, dt, screen, path=os.path.join('videos', 'frames')):
        self.dt = dt
        self.prev_t = 0.
        self.frame = 0
        self.screen = screen
        self.path = path
        if not os.path.exists(path):
            os.mkdir(path)

    def record(self, t):
        if t - self.prev_t >= self.dt:
            pygame.image.save(self.screen,
                              os.path.join(self.path,
                                           '{}.bmp'.format(self.frame)))
            self.frame += 1
            self.prev_t += self.dt


def cart_to_polar(cart_vec, positive=True):
    r = cart_vec.norm()
    theta = torch.cat([torch.atan2(cart_vec[Indices.Y], cart_vec[Indices.X])])
    if theta.data[0] < 0 and positive:
        theta += 2 * math.pi
    return r, theta


def polar_to_cart(r, theta):
    return torch.cat([torch.cos(theta), torch.sin(theta)]) * r


def cross_2d(v1, v2):
    return v1[0] * v2[1] - v1[1] * v2[0]


def plot(y_axis, x_axis=None):
    import matplotlib.pyplot as plt
    if x_axis is None:
        x_axis = range(len(y_axis))
    else:
        x_axis = [x.data[0] if x.__class__ is Variable else x[0] for x in x_axis]
    y_axis = [y.data[0] if y.__class__ is Variable else y[0] for y in y_axis]
    plt.plot(x_axis, y_axis)
    plt.show()


def get_instance(mod, class_id):
    """Checks if class_id is a string and if so loads class from module;
        else, just instantiates the class."""
    if isinstance(class_id, str):
        # Get by name if string
        return getattr(mod, class_id)()
    else:
        # Else just instantiate
        return class_id()
