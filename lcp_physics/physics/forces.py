from torch.autograd import Variable

from .utils import Params, wrap_variable


Tensor = Params.TENSOR_TYPE


def gravity(t):
    return ExternalForce.DOWN


def vert_impulse(t):
    if t < 0.1:
        return ExternalForce.DOWN
    else:
        return ExternalForce.ZEROS


def hor_impulse(t):
    if t < 0.2:
        return ExternalForce.RIGHT
    else:
        return ExternalForce.ZEROS


def rot_impulse(t):
    if t < 0.2:
        return ExternalForce.ROT
    else:
        return ExternalForce.ZEROS


class ExternalForce:
    # Pre-store basic forces
    DOWN = Variable(Tensor([0, 0, 1]))
    RIGHT = Variable(Tensor([0, 1, 0]))
    ROT = Variable(Tensor([1, 0, 0]))
    ZEROS = Variable(Tensor([0, 0, 0]))

    def __init__(self, force_func=gravity, multiplier=100.):
        self.multiplier = wrap_variable(multiplier)
        self.force = lambda t: force_func(t) * self.multiplier
