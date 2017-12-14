"""
Author: Filipe de Avila Belbute Peres
Based on: M. B. Cline, Rigid body simulation with contact and constraints, 2002
"""

import torch
from torch.autograd import Variable

from scipy.sparse.csc import csc_matrix
from scipy.sparse.linalg.dsolve.linsolve import splu, spsolve
import numpy as np

from .utils import Params
from lcp_physics.lcp.lcp import LCPFunction
from lcp_physics.lcp.solvers.batch_pdipm import forward, KKTSolvers, pre_factor_kkt


Tensor = Params.TENSOR_TYPE


class Engine:
    def __init__(self):
        pass

    def solve_dynamics(self, world, dt, stabilization=False):
        raise NotImplementedError


class PdipmEngine(Engine):
    def __init__(self):
        self.lcp_solver = LCPFunction

    def solve_dynamics(self, world, dt, stabilization=False):
        t = world.t
        # Get Jacobians
        Je = world.Je()
        Jc = None
        neq = Je.size(0) if Je.ndimension() > 0 else 0

        f = world.apply_forces(t)
        u = torch.matmul(world.M, world.v) + dt * f
        if neq > 0:
            u = torch.cat([u, Variable(Tensor(neq).zero_())])
        if not world.collisions:
            # No contact constraints, no need to solve LCP
            P = world.M
            if neq > 0:
                P = torch.cat([torch.cat([P, -Je.t()], dim=1),
                               torch.cat([Je, Variable(Tensor(neq, neq).zero_())],
                                         dim=1)])
            try:
                x = torch.matmul(torch.inverse(P), u)  # Eq. 2.41
            except RuntimeError:  # XXX
                print('\nRegularizing singular matrix.\n')
                x = torch.matmul(torch.inverse(P + Variable(torch.eye(P.size(0), P.size(1)).type_as(P.data) * 1e-10)), u)
        else:
            # Solve Mixed LCP (Kline 2.7.2)
            # TODO Organize
            Jc = world.Jc()
            v = torch.matmul(Jc, world.v * world.restitutions / 2)  # XXX why is 1/2 correction needed?
            TM = world.M.unsqueeze(0)
            if neq > 0:
                TJe = Je.unsqueeze(0)
                b = Variable(Tensor(Je.size(0)).unsqueeze(0).zero_())
            else:
                TJe = Variable(Tensor())
                b = Variable(None)
            TJc = Jc.unsqueeze(0) / 2
            Tu = u[:world.M.size(0)].unsqueeze(0)
            Tv = v.unsqueeze(0)
            Q_LU = S_LU = R = None
            # Q_LU, S_LU, R = pre_factor_kkt(TM, TJc, TJe)
            #######
            E = world.E()
            mu = world.mu()
            Jf = world.Jf()
            TJf = Jf.unsqueeze(0)
            TE = E.unsqueeze(0)
            Tmu = mu.unsqueeze(0)
            G = torch.cat([TJc, TJf,
                           Variable(Tensor(TJf.size(0), Tmu.size(1), TJf.size(2))
                                    .zero_())], dim=1)
            F = Variable(Tensor(G.size(1), G.size(1)).zero_().unsqueeze(0))
            F[:, TJc.size(1):-TE.size(2), -TE.size(2):] = TE
            F[:, -Tmu.size(1):, :Tmu.size(2)] = Tmu
            F[:, -Tmu.size(1):, Tmu.size(2):Tmu.size(2) + TE.size(1)] = \
                -TE.transpose(1, 2)
            h = torch.cat([Tv,
                           Variable(Tensor(Tv.size(0), TJf.size(1) + Tmu.size(1))
                                    .zero_())], 1)
            # adjust precision depending on difficulty of step, with maxIter in [3, 20]
            # measured by number of iterations performed on current step (world.dt / dt)
            max_iter = max(int(20 / (world.dt / dt)), 3)
            x = -self.lcp_solver(maxIter=max_iter)(TM, Tu, G, h, TJe, b, F)

        new_v = x[:world.vec_len * len(world.bodies)].squeeze(0)

        # Post-stabilization
        if stabilization:
            ge = torch.matmul(Je, new_v)
            if Jc is not None:
                gc = torch.matmul(Jc, new_v) + torch.matmul(Jc, new_v * -world.restitutions)
            else:
                gc = None
            dp = self.post_stabilization(world.M, Je, Jc, ge, gc)
            new_v = (new_v - dp).squeeze(0)  # XXX Is sign correct?
        return new_v

    def post_stabilization(self, M, Je, Jc, ge, gc):
        u = torch.cat([Variable(Tensor(Je.size(1)).zero_()), ge])
        if Jc is None:
            P = torch.cat([torch.cat([M, Je.t()], dim=1),
                           torch.cat([Je, Variable(Tensor(Je.size(0), Je.size(0)).zero_())],
                                     dim=1)])
            try:
                x = torch.matmul(torch.inverse(P), u)
            except RuntimeError:  # XXX
                print('\nRegularizing singular matrix in stabilization.\n')
                x = torch.matmul(torch.inverse(P + Variable(torch.eye(P.size(0), P.size(1)).type_as(P.data) * 1e-10)), u)
        else:
            v = gc
            TM = M.unsqueeze(0)
            TJe = Je.unsqueeze(0)
            TJc = Jc.unsqueeze(0)
            Th = u[:M.size(0)].unsqueeze(0)
            Tb = u[M.size(0):].unsqueeze(0)
            Tv = v.unsqueeze(0)
            F = Variable(Tensor(TJc.size(1), TJc.size(1)).zero_().unsqueeze(0))
            Q_LU = S_LU = R = None
            # Q_LU, S_LU, R = pre_factor_kkt(TM, TJc, TJe)
            x = self.lcp_solver()(TM, Th, TJc, Tv, TJe, Tb, F)
        # x = np.asarray(x).ravel()
        dp = x[:M.size(0)]
        return dp
