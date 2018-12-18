from .utils import get_tensor


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
    DOWN = get_tensor([0, 0, 1])
    RIGHT = get_tensor([0, 1, 0])
    ROT = get_tensor([1, 0, 0])
    ZEROS = get_tensor([0, 0, 0])

    def __init__(self, force_func=down_force, multiplier=100.):
        self.multiplier = multiplier
        self.force = lambda t: force_func(t) * self.multiplier
        self.body = None

    def set_body(self, body):
        self.body = body
        # match body's tensor type and device
        self.multiplier = get_tensor(self.multiplier, base_tensor=body._base_tensor)


class Gravity(ExternalForce):
    """Gravity force object, constantly returns a downwards pointing force of
       magnitude body.mass * g.
    """

    def __init__(self, g=10.0):
        self.multiplier = g
        self.body = None
        self.cached_force = None

    def force(self, t):
        return self.cached_force

    def set_body(self, body):
        super().set_body(body)
        down_tensor = ExternalForce.DOWN.type_as(body._base_tensor).to(body._base_tensor)
        self.cached_force = down_tensor * self.body.mass * self.multiplier
