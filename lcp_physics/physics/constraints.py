import pygame

import torch
from torch.autograd import Variable

from .utils import Indices, Params, cart_to_polar, polar_to_cart


X = Indices.X
Y = Indices.Y
DIM = Params.DIM

Tensor = Params.TENSOR_TYPE


class Joint:
    def __init__(self, body1, body2, pos):
        self.body1 = body1
        self.body2 = body2
        self.pos = Variable(Tensor(pos))
        self.pos1 = self.pos - self.body1.pos
        self.r1, self.rot1 = cart_to_polar(self.pos1)
        self.rot2 = None
        if body2 is not None:
            self.pos2 = self.pos - self.body2.pos
            self.r2, self.rot2 = cart_to_polar(self.pos2)

    def J(self):
        J1 = torch.cat([torch.cat([-self.pos1[Y], self.pos1[X]]).unsqueeze(1),
                        Variable(torch.eye(DIM).type_as(self.pos.data))], dim=1)
        if self.body2 is not None:
            J2 = torch.cat([torch.cat([self.pos2[Y], -self.pos2[X]]).unsqueeze(1),
                            -Variable(torch.eye(DIM).type_as(self.pos.data))], dim=1)
        else:
            J2 = Variable(Tensor(DIM, DIM + 1).zero_())
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
                                   self.pos.data.numpy().astype(int), 1)]
