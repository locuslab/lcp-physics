import torch
from torch.autograd import Function

from .solvers import pdipm
from .util import bger, extract_batch_size


class LCPFunction(Function):
    """A differentiable LCP solver, uses the primal dual interior point method
       implemented in pdipm.
    """
    def __init__(self, eps=1e-12, verbose=-1, not_improved_lim=3,
                 max_iter=10):
        super().__init__()
        self.eps = eps
        self.verbose = verbose
        self.not_improved_lim = not_improved_lim
        self.max_iter = max_iter
        self.Q_LU = self.S_LU = self.R = None

    # @profile
    def forward(self, Q, p, G, h, A, b, F):
        _, nineq, nz = G.size()
        neq = A.size(1) if A.ndimension() > 1 else 0
        assert(neq > 0 or nineq > 0)
        self.neq, self.nineq, self.nz = neq, nineq, nz

        self.Q_LU, self.S_LU, self.R = pdipm.pre_factor_kkt(Q, G, F, A)
        zhats, self.nus, self.lams, self.slacks = pdipm.forward(
            Q, p, G, h, A, b, F, self.Q_LU, self.S_LU, self.R,
            eps=self.eps, max_iter=self.max_iter, verbose=self.verbose,
            not_improved_lim=self.not_improved_lim)

        self.save_for_backward(zhats, Q, p, G, h, A, b, F)
        return zhats

    def backward(self, dl_dzhat):
        zhats, Q, p, G, h, A, b, F = self.saved_tensors
        batch_size = extract_batch_size(Q, p, G, h, A, b)

        neq, nineq, nz = self.neq, self.nineq, self.nz

        # D = torch.diag((self.lams / self.slacks).squeeze(0)).unsqueeze(0)
        d = self.lams / self.slacks

        pdipm.factor_kkt(self.S_LU, self.R, d)
        dx, _, dlam, dnu = pdipm.solve_kkt(self.Q_LU, d, G, A, self.S_LU,
                                           dl_dzhat, G.new_zeros(batch_size, nineq),
                                           G.new_zeros(batch_size, nineq),
                                           G.new_zeros(batch_size, neq))

        dps = dx
        dGs = (bger(dlam, zhats) + bger(self.lams, dx))
        dFs = bger(dlam, self.lams)
        dhs = -dlam
        if neq > 0:
            dAs = bger(dnu, zhats) + bger(self.nus, dx)
            dbs = -dnu
        else:
            dAs, dbs = None, None
        dQs = 0.5 * (bger(dx, zhats) + bger(zhats, dx))

        grads = (dQs, dps, dGs, dhs, dAs, dbs, dFs)
        return grads

    def numerical_backward(self, dl_dzhat):
        # XXX experimental
        # adapted from pytorch's grad check
        # from torch.autograd.gradcheck import get_numerical_jacobian
        from torch.autograd import Variable
        from collections import Iterable

        def make_jacobian(x, num_out):
            if isinstance(x, Variable) and not x.requires_grad:
                return None
            elif torch.is_tensor(x) or isinstance(x, Variable):
                return torch.zeros(x.nelement(), num_out)
            elif isinstance(x, Iterable):
                jacobians = list(filter(
                    lambda x: x is not None, (make_jacobian(elem, num_out) for elem in x)))
                if not jacobians:
                    return None
                return type(x)(jacobians)
            else:
                return None

        def iter_tensors(x, only_requiring_grad=False):
            if torch.is_tensor(x):
                yield x
            elif isinstance(x, Variable):
                if x.requires_grad or not only_requiring_grad:
                    yield x.data
            elif isinstance(x, Iterable):
                for elem in x:
                    for result in iter_tensors(elem, only_requiring_grad):
                        yield result

        def contiguous(x):
            if torch.is_tensor(x):
                return x.contiguous()
            elif isinstance(x, Variable):
                return x.contiguous()
            elif isinstance(x, Iterable):
                return type(x)(contiguous(e) for e in x)
            return x

        def get_numerical_jacobian(fn, inputs, target, eps=1e-3):
            # To be able to use .view(-1) input must be contiguous
            inputs = contiguous(inputs)
            target = contiguous(target)
            output_size = fn(*inputs).numel()
            jacobian = make_jacobian(target, output_size)

            # It's much easier to iterate over flattened lists of tensors.
            # These are reference to the same objects in jacobian, so any changes
            # will be reflected in it as well.
            x_tensors = [t for t in iter_tensors(target, True)]
            j_tensors = [t for t in iter_tensors(jacobian)]

            outa = torch.DoubleTensor(output_size)
            outb = torch.DoubleTensor(output_size)

            # TODO: compare structure
            for x_tensor, d_tensor in zip(x_tensors, j_tensors):
                flat_tensor = x_tensor.view(-1)
                for i in range(flat_tensor.nelement()):
                    orig = flat_tensor[i]
                    flat_tensor[i] = orig - eps
                    outa.copy_(fn(*inputs), broadcast=False)
                    flat_tensor[i] = orig + eps
                    outb.copy_(fn(*inputs), broadcast=False)
                    flat_tensor[i] = orig

                    outb.add_(-1, outa).div_(2 * eps)
                    d_tensor[i] = outb

            return jacobian

        zhats = self.saved_tensors[0]
        inputs = self.saved_tensors[1:]
        grads = []
        for x in inputs:
            dl_dx = None
            if len(x.size()) > 0:
                jacobian = get_numerical_jacobian(self.forward, inputs, target=x.squeeze(0),
                                                  eps=1e-5).type_as(dl_dzhat)
                dl_dx = jacobian.matmul(dl_dzhat.t()).view(x.size())
            grads.append(dl_dx)
        # grads = (dQs, dps, dGs, dhs, dAs, dbs, dFs)
        # grads_compare = self.analytical_backward(dl_dzhat)
        return tuple(grads)
