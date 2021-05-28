import numpy as np
from functools import lru_cache
import torch
import torch.fft


@lru_cache(maxsize=1000)
def count_varible(xmax, dtype, device):
    x = torch.arange(xmax, dtype=dtype, device=device)
    return x


@lru_cache(maxsize=1000)
def get_idxs(n1, n2, n3, n_ker, ker_shift, device):
    idxs = n1 * (-ker_shift + torch.arange(n_ker, n3, device=device)[:, None, None] - torch.arange(n_ker, device=device)[None, :, None]) \
           + n2 * (torch.arange(n_ker, device=device))[None, :, None] + torch.arange(n2, device=device)
    return idxs


@lru_cache()
def get_complex_pairs(dtype, device):
    r = torch.complex(torch.tensor(1, dtype=dtype, device=device), torch.tensor(0, dtype=dtype, device=device))
    i = torch.complex(torch.tensor(0, dtype=dtype, device=device), torch.tensor(1, dtype=dtype, device=device))
    return r, i

@lru_cache()
def lin_interp(n, dtype=torch.double, device='cpu'):
    return torch.linspace(0, 1, n, dtype=dtype, device=device)


def calc_p_days(timeseries_orig, timeseries, r0, k, cross, total_fraction, ker, ker_shift, dtype, device,
                simulation=False):
    # Speed: a lot could be won by having non-linear xmax
    xmax = torch.max(timeseries_orig.long()) + 2  # the highest count seen + 1

    x = count_varible(xmax, dtype, device)

    n_ker = ker.shape[-1]

    # Redefine r0 and k to be used in NB formula
    if simulation:
        cross_timeseries = (1e-10 + cross * total_fraction[:, None, None] + timeseries[:, None, None])
    else:
        cross_timeseries = (1e-10 + cross * total_fraction[:, None, None] + (1 - cross) * timeseries[:, None, None])

    r0 = r0 * cross_timeseries * ker[:, :, None]
    k = k * cross_timeseries
    x = x[None, None, :]

    # Calculate NB probabilities
    log_P = k * torch.log(k) + x * torch.log(r0) - (x + k) * torch.log(k + r0) + torch.lgamma(k + x) - torch.lgamma(k) - torch.lgamma(1 + x)
    P = torch.exp(log_P)

    # Now do convolutions ala: (here writing example with ker_shift = -1)
    # P_day[n, :] = P[n + 1, 0, :] * P[n, 1, :] * P[n - 1, 2, :] * P[n - 2, 3, :] * P[n - 3, 4, :] ...
    # (to do this we use flat indices and `torch.take`)

    n1 = P.shape[1] * P.shape[2]
    n2 = P.shape[2]
    n3 = P.shape[0]

    idxs = get_idxs(n1, n2, n3, n_ker, ker_shift, device)

    if ker_shift < 0:
        P = torch.cat((P, P[-1, :, :].repeat(-ker_shift, 1, 1)), dim=0)
    P = torch.take(P, idxs)

    # Now do convolution:
    t = P[:, 0, :]
    for i in range(1, P.shape[1]):
        w = torch.flip(P[:, i, :], dims=(1,))
        t = torch.conv1d(t[None, :, :], w[:, None, :], groups=t.shape[0],
                         padding=t.shape[1] - 1)[0, :, :t.shape[1]]

    P = t

    # Find probability of observations
    if timeseries_orig.dtype == torch.long:
        p_days = P[torch.arange(len(timeseries_orig)), timeseries_orig]
    else:  # Linear interpolation
        ii = torch.arange(len(timeseries_orig), device=timeseries_orig.device)
        x = x[0, 0, :]
        idxs = timeseries_orig.long() + 1
        left = P[ii, idxs - 1]
        right = P[ii, idxs]
        s = (timeseries_orig - x[idxs - 1])
        p_days = left + s * (right - left)

    return p_days


def _binomial_log_likelihood(timeseries, observed, p):
    n = timeseries
    k = observed
    return log_binom(n, k) + k * torch.log(p) + (n - k) * torch.log(1 - p)


def binomial_log_likelihood(timeseries, observed, p):
    return torch.sum(_binomial_log_likelihood(timeseries, observed, p))


def log_likelihood(timeseries_orig, timeseries, r0, k, cross, total_fraction, ker, ker_shift, dtype, device,
                   start_index=0, simulation=False, end_index=-1, weights=None):
    p_days = calc_p_days(timeseries_orig, timeseries, r0, k, cross, total_fraction, ker, ker_shift, dtype, device,
                         simulation=simulation)[start_index:end_index]

    if weights is not None:
        return torch.sum(weights[start_index:end_index] * torch.log(p_days))
    else:
        return torch.sum(torch.log(p_days))


def torch_binom(n, k):
    mask = n.detach() >= k.detach()
    n = mask * n
    k = mask * k
    a = torch.lgamma(n + 1) - torch.lgamma((n - k) + 1) - torch.lgamma(k + 1)
    return torch.exp(a) * mask


def log_binom(n, k):
    return torch.lgamma(n + 1) - torch.lgamma((n - k) + 1) - torch.lgamma(k + 1)


def log_factorial(n):
    return torch.lgamma(n + 1)


def log_pochhammer(x, n):
    return torch.lgamma(x + n) - torch.lgamma(x)


