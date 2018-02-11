import pygame

import torch
from torch.autograd import Variable

from .utils import Indices, Params, wrap_variable, cart_to_polar, polar_to_cart


X = Indices.X
Y = Indices.Y
DIM = Params.DIM

Tensor = Params.TENSOR_TYPE


class Joint:
    def __init__(self, body1, body2, pos):
        self.num_constraints = 2
        self.body1 = body1
        self.body2 = body2
        self.pos = wrap_variable(pos)
        self.pos1 = self.pos - self.body1.pos
        self.r1, self.rot1 = cart_to_polar(self.pos1)
        self.rot2 = None
        if body2 is not None:
            self.pos2 = self.pos - self.body2.pos
            self.r2, self.rot2 = cart_to_polar(self.pos2)

    def J(self):
        J1 = torch.cat([torch.cat([-self.pos1[Y], self.pos1[X]]).unsqueeze(1),
                        Variable(torch.eye(DIM).type_as(self.pos.data))], dim=1)
        J2 = None
        if self.body2 is not None:
            J2 = torch.cat([torch.cat([self.pos2[Y], -self.pos2[X]]).unsqueeze(1),
                            -Variable(torch.eye(DIM).type_as(self.pos.data))], dim=1)
        return J1, J2

    def move(self, dt):
        self.rot1 = self.rot1 + self.body1.v[0] * dt
        if self.body2 is not None:
            self.rot2 = self.rot2 + self.body2.v[0] * dt
        self.update_pos()

    def update_pos(self):
        self.pos1 = polar_to_cart(self.r1, self.rot1)
        self.pos = self.body1.pos + self.pos1
        if self.body2 is not None:
            # keep position on body1 as reference
            self.pos2 = self.pos - self.body2.pos

    def draw(self, screen):
        return [pygame.draw.circle(screen, (0, 255, 0),
                                   self.pos.data.numpy().astype(int), 2)]


class YConstraint:
    def __init__(self, body1):
        self.num_constraints = 1
        self.body1 = body1
        self.pos = body1.pos
        self.pos1 = self.pos - self.body1.pos
        self.r1, self.rot1 = cart_to_polar(self.pos1)

        self.body2 = self.rot2 = None

    def J(self):
        J = Variable(Tensor([0, 0, 1])).type_as(self.pos.data).unsqueeze(0)
        return J, None

    def move(self, dt):
        self.rot1 = self.rot1 + self.body1.v[0] * dt
        self.update_pos()

    def update_pos(self):
        self.pos1 = polar_to_cart(self.r1, self.rot1)
        self.pos = self.body1.pos + self.pos1

    def draw(self, screen):
        pos = self.pos.data.numpy().astype(int)
        return [pygame.draw.line(screen, (0, 255, 0), pos - [5, 0], pos + [5, 0], 2)]


class XConstraint:
    def __init__(self, body1):
        self.num_constraints = 1
        self.body1 = body1
        self.pos = body1.pos
        self.pos1 = self.pos - self.body1.pos
        self.r1, self.rot1 = cart_to_polar(self.pos1)

        self.body2 = self.rot2 = None

    def J(self):
        J = Variable(Tensor([0, 1, 0])).type_as(self.pos.data).unsqueeze(0)
        return J, None

    def move(self, dt):
        self.rot1 = self.rot1 + self.body1.v[0] * dt
        self.update_pos()

    def update_pos(self):
        self.pos1 = polar_to_cart(self.r1, self.rot1)
        self.pos = self.body1.pos + self.pos1

    def draw(self, screen):
        pos = self.pos.data.numpy().astype(int)
        return [pygame.draw.line(screen, (0, 255, 0), pos - [0, 5], pos + [0, 5], 2)]


class RotConstraint:
    def __init__(self, body1):
        self.num_constraints = 1
        self.body1 = body1
        self.pos = body1.pos
        self.pos1 = self.pos - self.body1.pos
        self.r1, self.rot1 = cart_to_polar(self.pos1)

        self.body2 = self.rot2 = None

    def J(self):
        J = Variable(Tensor([1, 0, 0])).type_as(self.pos.data).unsqueeze(0)
        return J, None

    def move(self, dt):
        self.rot1 = self.rot1 + self.body1.v[0] * dt
        self.update_pos()

    def update_pos(self):
        self.pos1 = polar_to_cart(self.r1, self.rot1)
        self.pos = self.body1.pos + self.pos1

    def draw(self, screen):
        return [pygame.draw.circle(screen, (0, 255, 0),
                                   self.pos.data.numpy().astype(int),
                                   5, 1)]


class TotalConstraint:
    def __init__(self, body1):
        self.num_constraints = 3
        self.body1 = body1
        self.pos = body1.pos
        self.pos1 = self.pos - self.body1.pos
        self.r1, self.rot1 = cart_to_polar(self.pos1)

        self.body2 = self.rot2 = None
        self.eye = torch.eye(self.num_constraints).type_as(self.pos.data)

    def J(self):
        J = Variable(self.eye)
        return J, None

    def move(self, dt):
        self.rot1 = self.rot1 + self.body1.v[0] * dt
        self.update_pos()

    def update_pos(self):
        self.pos1 = polar_to_cart(self.r1, self.rot1)
        self.pos = self.body1.pos + self.pos1

    def draw(self, screen):
        pos = self.pos.data.numpy().astype(int)
        return [pygame.draw.circle(screen, (0, 255, 0), pos + 1, 5, 1),
                pygame.draw.line(screen, (0, 255, 0), pos - [5, 0], pos + [5, 0], 2),
                pygame.draw.line(screen, (0, 255, 0), pos - [0, 5], pos + [0, 5], 2)]
