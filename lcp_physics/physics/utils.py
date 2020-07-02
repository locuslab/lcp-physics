import os
import math

import pygame

import torch


class Defaults:
    """Aggregates general simulation parameters defaults.
    """
    # Dimensions
    DIM = 2

    # Contact detectopm parameter
    EPSILON = 30

    # Penetration tolerance parameter
    TOL = 1e-6

    # Default simulation parameters
    RESTITUTION = 0.5

    FRIC_COEFF = 0.9
    FRIC_DIRS = 2

    FPS = 30
    DT = 1.0 / FPS

    ENGINE = 'PdipmEngine'
    CONTACT = 'DiffContactHandler'

    # Tensor defaults
    DTYPE = torch.double
    DEVICE = torch.device('cpu')
    LAYOUT = torch.strided

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
    """Records simulations into a series of image frames.
    """
    def __init__(self, dt, screen, path=os.path.join('videos', 'frames')):
        self.dt = dt
        self.prev_t = 0.
        self.frame = 0
        self.screen = screen
        self.path = path
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    def record(self, t):
        if t - self.prev_t >= self.dt:
            pygame.image.save(self.screen,
                              os.path.join(self.path,
                                           '{}.bmp'.format(self.frame)))
            self.frame += 1
            self.prev_t += self.dt


def cart_to_polar(cart_vec, positive=True):
    """Converts cartesian to polar coordinates.
    """
    r = cart_vec.norm()
    theta = torch.atan2(cart_vec[Indices.Y], cart_vec[Indices.X])
    if theta.item() < 0 and positive:
        theta = theta + 2 * math.pi
    return r, theta


def polar_to_cart(r, theta):
    """Converts polar to cartesian coordinates.
    """
    ret = torch.cat([torch.cos(theta).unsqueeze(0),
                     torch.sin(theta).unsqueeze(0)]).squeeze() * r
    return ret


def cross_2d(v1, v2):
    """Two dimensional cross product.
    """
    return v1[0] * v2[1] - v1[1] * v2[0]


def left_orthogonal(v):
    """Get the (left) orthogonal vector to the provided vector.
    """
    return torch.stack([v[1], -v[0]])


def rotation_matrix(ang):
    """Get the rotation matrix for a specific angle.
    """
    s, c = torch.sin(ang), torch.cos(ang)
    rot_mat = ang.new_empty(2, 2)
    rot_mat[0, 0] = rot_mat[1, 1] = c
    rot_mat[0, 1], rot_mat[1, 0] = -s, s
    return rot_mat


def get_tensor(x, base_tensor=None, **kwargs):
    """Wrap array or scalar in torch Tensor, if not already.
    """
    if isinstance(x, torch.Tensor):
        return x
    elif base_tensor is not None:
        return base_tensor.new_tensor(x, **kwargs)
    else:
        return torch.tensor(x,
                            dtype=Defaults.DTYPE,
                            device=Defaults.DEVICE,
                            # layout=Params.DEFAULT_LAYOUT,
                            **kwargs,
                            )


def plot(y_axis, x_axis=None):
    import matplotlib.pyplot as plt
    if x_axis is None:
        x_axis = range(len(y_axis))
    else:
        x_axis = [x.item() if x.__class__ is torch.Tensor else x for x in x_axis]
    y_axis = [y.item() if y.__class__ is torch.Tensor else y for y in y_axis]
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
