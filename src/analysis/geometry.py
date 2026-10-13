"""Weighted Kabsch alignment shared by training and sampling."""
import torch

def rmsdalign(a, b, weights=None):  # alignes B to A  # [*, N, 3]
    B = a.shape[:-2]
    N = a.shape[-2]
    if weights is None:
        weights = a.new_ones(*B, N)
    weights = weights.unsqueeze(-1)
    a_mean = (a * weights).sum(-2, keepdims=True) / weights.sum(-2, keepdims=True)
    a = a - a_mean
    b_mean = (b * weights).sum(-2, keepdims=True) / weights.sum(-2, keepdims=True)
    b = b - b_mean
    B = torch.einsum("...ji,...jk->...ik", weights * a, b)
    u, s, vh = torch.linalg.svd(B)

    # Correct improper rotation if necessary (as in Kabsch algorithm)
    """
    if torch.linalg.det(u @ vh) < 0:
        s[-1] = -s[-1]
        u[:, -1] = -u[:, -1]
    """
    sgn = torch.sign(torch.linalg.det(u @ vh))
    s[..., -1] *= sgn
    u[..., :, -1] *= sgn.unsqueeze(-1)
    C = u @ vh  # c rotates B to A
    return b @ C.mT + a_mean
