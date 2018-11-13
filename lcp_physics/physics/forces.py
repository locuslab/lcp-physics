from torch.autograd import Variable

from .utils import Params, wrap_variable


Tensor = Params.TENSOR_TYPE


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
    DOWN = Variable(Tensor([0, 0, 1]))
    RIGHT = Variable(Tensor([0, 1, 0]))
    ROT = Variable(Tensor([1, 0, 0]))
    ZEROS = Variable(Tensor([0, 0, 0]))

    def __init__(self, force_func=down_force, multiplier=100.):
        self.multiplier = wrap_variable(multiplier)
        self.force = lambda t: force_func(t) * self.multiplier
        self.body = None


class Gravity(ExternalForce):
    """Gravity force object, constantly returns a downwards pointing force of
       magnitude body.mass * g.
    """
    def __init__(self, g=10.0):
        self.multiplier = wrap_variable(g)
        self.body = None

    def force(self, t):
        return ExternalForce.DOWN * self.body.mass * self.multiplier
