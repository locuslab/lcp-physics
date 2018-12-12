import torch
from .utils import Params, wrap_tensor


DTYPE = Params.TENSOR_TYPE


def down_force(t):
    return ExternalForce.DOWN


def vert_impulse(t):
    if t < 0.1:
        return ExternalForce.DOWN
    else:
        return ExternalForce.ZEROS


def hor_impulse(t):
    if t < 0.1:
        return ExternalForce.RIGHT
    else:
        return ExternalForce.ZEROS


def rot_impulse(t):
    if t < 0.1:
        return ExternalForce.ROT
    else:
        return ExternalForce.ZEROS


class ExternalForce:
    """Generic external force to be added to objects.
       Takes in a force_function which returns a force vector as a function of time,
       and a multiplier that multiplies such vector.
    """
    # Pre-store basic forces
    DOWN = torch.tensor([0, 0, 1], dtype=DTYPE)
    RIGHT = torch.tensor([0, 1, 0], dtype=DTYPE)
    ROT = torch.tensor([1, 0, 0], dtype=DTYPE)
    ZEROS = torch.tensor([0, 0, 0], dtype=DTYPE)

    def __init__(self, force_func=down_force, multiplier=100.):
        self.multiplier = wrap_tensor(multiplier)
        self.force = lambda t: force_func(t) * self.multiplier
        self.body = None


class Gravity(ExternalForce):
    """Gravity force object, constantly returns a downwards pointing force of
       magnitude body.mass * g.
    """
    def __init__(self, g=10.0):
        self.multiplier = wrap_tensor(g)
        self.body = None

    def force(self, t):
        return ExternalForce.DOWN * self.body.mass * self.multiplier
