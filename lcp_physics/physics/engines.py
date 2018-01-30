"""
Author: Filipe de Avila Belbute Peres
Based on: M. B. Cline, Rigid body simulation with contact and constraints, 2002
"""

import torch
from torch.autograd import Variable

from scipy.sparse.csc import csc_matrix
from scipy.sparse.linalg.dsolve.linsolve import splu, spsolve
import numpy as np

from .utils import Params, binverse
from lcp_physics.lcp.lcp import LCPFunction


Tensor = Params.TENSOR_TYPE


class Engine:
    def __init__(self):
        pass

    def solve_dynamics(self, world, dt, stabilization=False):
        raise NotImplementedError

    def batch_solve_dynamics(self, world, dt, stabilization=False):
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
        u = torch.matmul(world.M(), world.get_v()) + dt * f
        if neq > 0:
            u = torch.cat([u, Variable(Tensor(neq).zero_())])
        if not world.collisions:
            # No contact constraints, no need to solve LCP
            if neq > 0:
                P = torch.cat([torch.cat([world.M(), -Je.t()], dim=1),
                               torch.cat([Je, Variable(Tensor(neq, neq).zero_())],
                                         dim=1)])
                try:
                    x = torch.matmul(torch.inverse(P), u)  # Eq. 2.41
                except RuntimeError:  # XXX
                    # print('\nRegularizing singular matrix.\n')
                    x = torch.matmul(torch.inverse(P + Variable(torch.eye(P.size(0),
                            P.size(1)).type_as(P.data) * 1e-7)), u)
            else:
                x = torch.matmul(world.invM(), u)  # Eq. 2.41
        else:
            # Solve Mixed LCP (Kline 2.7.2)
            # TODO Organize
            Jc = world.Jc() #/ 2  # / 2 correction is needed for some reason
            v = torch.matmul(Jc, world.get_v() * world.restitutions)
            TM = world.M().unsqueeze(0)
            if neq > 0:
                TJe = Je.unsqueeze(0)
                b = Variable(Tensor(Je.size(0)).unsqueeze(0).zero_())
            else:
                TJe = Variable(Tensor())
                b = Variable(None)
            TJc = Jc.unsqueeze(0)
            Tu = u[:world.M().size(0)].unsqueeze(0)
            Tv = v.unsqueeze(0)
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
            x = -self.lcp_solver(maxIter=max_iter, verbose=-1)(TM, Tu, G, h, TJe, b, F)

        new_v = x[:world.vec_len * len(world.bodies)].squeeze(0)

        # Post-stabilization
        if stabilization:
            ge = torch.matmul(Je, new_v)
            if Jc is not None:
                gc = torch.matmul(Jc, new_v) + torch.matmul(Jc, new_v * -world.restitutions)
            dp = self.post_stabilization(world.M(), Je, Jc, ge, gc)
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
            x = self.lcp_solver()(TM, Th, TJc, Tv, TJe, Tb, F)
        # x = np.asarray(x).ravel()
        dp = x[:M.size(0)]
        return dp

    def batch_solve_dynamics(self, world, dt, stabilization=False):
        t = world.t
        f = world.apply_forces(t)
        u_ = torch.bmm(world.M(), world.get_v().unsqueeze(2)).squeeze(2) + dt * f
        x = Variable(world.get_v().data.new(world.get_v().size()).zero_())
        colls_idx = world.has_n_collisions(0)
        if colls_idx.any():
            x_idx = colls_idx.unsqueeze(1).expand(x.size(0), x.size(1))
            M = world.M(num_colls=0)
            batch_size = M.size(0)
            Je = world.Je(num_colls=0)
            neq = Je.size(1) if Je.dim() > 0 else 0
            u = u_[colls_idx.unsqueeze(1).expand(colls_idx.size(0), u_.size(1))].view(torch.sum(colls_idx), -1)
            # No contact constraints, no need to solve LCP
            A = M
            if neq > 0:
                u = torch.cat([u, Variable(Tensor(batch_size, neq).zero_())], 1)
                A = torch.cat([torch.cat([M, -Je.transpose(1, 2)], dim=2),
                               torch.cat([Je, Variable(Tensor(batch_size, neq, neq).zero_())],
                                         dim=2)], dim=1)
                try:
                    x[x_idx] = torch.bmm(binverse(A), u.unsqueeze(2)).squeeze(2)  # Eq. 2.41
                except RuntimeError:  # XXX
                    print('\nRegularizing singular matrix.\n')
                    # XXX Use expand below?
                    reg = Variable(torch.eye(A.size(1), A.size(2)).type_as(A.data) * 1e-7).repeat(A.size(0), 1, 1)
                    x[x_idx] = torch.bmm(binverse(A + reg), u.unsqueeze(2)).squeeze(2)
            else:
                # XXX Works only for immutable M matrices (true for circles and squares)
                x[x_idx] = torch.bmm(world.invM(num_colls=0), u.unsqueeze(2)).squeeze(2)

        # Solve Mixed LCP (Kline 2.7.2)
        # TODO Organize
        for i in range(1, 2):  # TODO Number of possible collisions
            colls_idx = world.has_n_collisions(i)
            if not colls_idx.any():
                continue
            x_idx = colls_idx.unsqueeze(1).expand(x.size(0), x.size(1))

            M = world.M(num_colls=i)
            u = u_[colls_idx.unsqueeze(1).expand(colls_idx.size(0), u_.size(1))].view(torch.sum(colls_idx), -1)
            Je = world.Je(num_colls=i)
            neq = Je.size(1) if Je.dim() > 0 else 0
            Jc = world.Jc(num_colls=i) #/ 2  # / 2 correction is needed for some reason
            v = torch.bmm(Jc, (world.get_v(num_colls=i) * world.restitutions(num_colls=i)).unsqueeze(2)).squeeze(2)
            if neq > 0:
                b = Variable(Tensor(batch_size, Je.size(1)).zero_())
            else:
                Je = Variable(Tensor())
                b = Variable(None)
            E = world.E(num_colls=i)
            mu = world.mu(num_colls=i)
            Jf = world.Jf(num_colls=i)

            G = torch.cat([Jc, Jf,
                           Variable(Tensor(Jf.size(0), mu.size(1), Jf.size(2))
                                    .zero_())], dim=1)
            F = Variable(Tensor(G.size(0), G.size(1), G.size(1)).zero_())

            F[:, Jc.size(1):-E.size(2), -E.size(2):] = E
            F[:, -mu.size(1):, :mu.size(2)] = mu
            F[:, -mu.size(1):, mu.size(2):mu.size(2) + E.size(1)] = \
                -E.transpose(1, 2)
            h = torch.cat([v,
                           Variable(Tensor(v.size(0), Jf.size(1) + mu.size(1))
                                    .zero_())], 1)

            # adjust precision depending on difficulty of step, with maxIter in [3, 20]
            # measured by number of iterations performed on current step (world.dt / dt)
            max_iter = max(int(20 / (world.dt / dt)), 3)
            x[x_idx] = -self.lcp_solver(maxIter=max_iter, verbose=-1)(M, u, G, h, Je, b, F)

        new_v = x[:, :world.vec_len * len(world.worlds[0].bodies)]

        # Post-stabilization
        if stabilization:
            raise NotImplementedError
        return new_v
