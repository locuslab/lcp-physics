import torch
import numpy as np


def print_header(msg):
    print('===>', msg)


def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.detach().cpu().numpy()


def bger(x, y):
    return x.unsqueeze(2).bmm(y.unsqueeze(1))


def get_sizes(G, A=None):
    if G.dim() == 2:
        nineq, nz = G.size()
        nBatch = 1
    elif G.dim() == 3:
        nBatch, nineq, nz = G.size()
    if A is not None:
        if A.ndimension() <= 1:
            neq = 0
        elif A.dim() == 2:
            neq = A.size(0)
        elif A.dim() == 3:
            neq = A.size(1)
    else:
        neq = None
    # nBatch = batchedTensor.size(0) if batchedTensor is not None else None
    return nineq, nz, neq, nBatch


def bdiag(d):
    nBatch, sz = d.size()
    D = torch.zeros(nBatch, sz, sz).type_as(d)
    I = torch.eye(sz).repeat(nBatch, 1, 1).type_as(d).byte()
    D[I] = d.squeeze()
    return D


def expandParam(X, nBatch, nDim):
    if X.ndimension() in (0, nDim):
        return X, False
    elif X.ndimension() == nDim - 1:
        return X.unsqueeze(0).expand(*([nBatch] + list(X.size()))), True
    else:
        raise RuntimeError("Unexpected number of dimensions.")


def extract_batch_size(Q, p, G, h, A, b):
    dims = [3, 2, 3, 2, 3, 2]
    params = [Q, p, G, h, A, b]
    for param, dim in zip(params, dims):
        if param.ndimension() == dim:
            return param.size(0)
    return 1


def efficient_btriunpack(LU_data, LU_pivots, unpack_data=True, unpack_pivots=True):
    """More efficient version of torch.btriunpack.

    From https://github.com/pytorch/pytorch/issues/15182
    """
    nBatch, sz = LU_data.shape[:-1]

    if unpack_data:
        I_U = torch.ones(sz, sz, device=LU_data.device, dtype=torch.uint8).triu_().expand_as(LU_data)
        zero = torch.tensor(0.).type_as(LU_data)
        U = torch.where(I_U, LU_data, zero)
        L = torch.where(I_U, zero, LU_data)
        L.diagonal(dim1=-2, dim2=-1).fill_(1)
    else:
        L = U = None

    if unpack_pivots:
        P = torch.eye(sz, device=LU_data.device, dtype=LU_data.dtype).unsqueeze(0).repeat(nBatch, 1, 1)
        LU_pivots = LU_pivots - 1
        for i in range(nBatch):
            final_order = list(range(sz))
            for k, j in enumerate(LU_pivots[i]):
                final_order[k], final_order[j] = final_order[j], final_order[k]
            P[i] = P[i][final_order]
        P = P.transpose(-2, -1)  # This is because we permuted the rows in the previous operation
    else:
        P = None

    return P, L, U
