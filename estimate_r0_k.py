import pickle
from functools import lru_cache
import torch
from numba import jit, prange
import torch.fft
import scipy.optimize
import numpy as np
import scipy.interpolate
from scipy.signal import savgol_filter
from model import log_likelihood
import argparse


parser = argparse.ArgumentParser(prog='SuperBayesian')

parser.add_argument('--simulation-file', type=str, default='simulation.pkl')
parser.add_argument('--cpu', action="store_true")
parser.add_argument('--sampler', type=str, default='optimize')
parser.add_argument('--sample-cross', action="store_true")
parser.add_argument('--start-index', type=int, default=0)
parser.add_argument('--end-index', type=int, default=-1)
parser.add_argument('--sampler-n', type=int, default=1000)
parser.add_argument('--cross', type=float, default=0.05)
parser.add_argument('--save-as', type=str)
parser.add_argument('--r0-samples', type=int, default=0)
parser.add_argument('--infect-mean', type=float, default=5)
parser.add_argument('--infect-var', type=float, default=10)
parser.add_argument('--discover-mean', type=float, default=4.5)
parser.add_argument('--discover-var', type=float, default=5)


args = parser.parse_args()
args.simulation = True


def estimate_R0(case_arr, time):
    total = case_arr.sum(axis=0)

    smooth = np.exp(savgol_filter(np.log(total + 1e-10), 51, 3))
    smooth = np.exp(savgol_filter(np.log(smooth + 1e-10), 51, 3))
    smooth_d = scipy.interpolate.PchipInterpolator(time, smooth).derivative(1)(time)
    gamma = 1 / 3.
    R0 = 1 + smooth_d / smooth / gamma
    return R0


def gamma_dist(x, r0, k):
    p = np.exp(-(k * x) / r0) * (r0/k)**(-k) * x**(k - 1) / scipy.special.gamma(k)
    p[p < 0] = 0
    return p


@lru_cache()
def likelihood_kernel(dtype, device):
    x = np.arange(15)
    p_infect = gamma_dist(x, args.infect_mean, args.infect_var)
    p_infect = p_infect / p_infect.sum()
    p_discover = gamma_dist(x, args.discover_mean, args.discover_var)
    p_discover = p_discover / p_discover.sum()

    # Convolve:
    p = np.convolve(np.convolve(p_infect, p_discover), p_discover[::-1])
    t = np.arange(len(p_discover) - (len(p_infect) + len(p_discover) - 2) - 1, len(p_infect) + len(p_discover) - 1)

    mask = p > 0.001
    p = p[mask]
    t = t[mask]
    p /= p.sum()

    shift = t[0]

    return torch.tensor(p, dtype=dtype, device=device), torch.tensor(p_infect, dtype=dtype, device=device),\
           torch.tensor(p_discover, dtype=dtype, device=device), shift


@jit(nopython=True, parallel=True)
def _adapt_count(tmin, tmax, samples, counts):
    for i in prange(samples.shape[0]):
        for j in range(samples.shape[1]):
            if tmin <= samples[i, j] and samples[i, j] <= tmax:
                counts[i, samples[i, j] - tmin] += 1


def _single_adapt_to_tests(t, test_arr, p_infect, p_discover, n_samples):
    max_before = len(p_discover)
    padded_test_arr = np.pad(test_arr, (max_before, max_before), mode='edge')

    ni = len(p_infect)
    nd = len(p_discover)
    n_infect = np.arange(ni)
    n_discover = np.arange(nd)

    samples = np.zeros((len(test_arr), n_samples), dtype=int) - 100
    _adapt_tests_loop(max_before, n_discover, n_infect, n_samples, nd, p_discover, p_infect, padded_test_arr,
                      samples, test_arr)

    counts = np.zeros((samples.shape[0], len(t)), dtype=int)
    _adapt_count(min(t), max(t), samples, counts)

    p = counts / counts.sum(axis=1)[:, None]

    return p


@jit(nopython=True)
def rand_choice_nb(arr, p):
    return arr[np.searchsorted(np.cumsum(p), np.random.random(), side="right")]


@jit(nopython=True, parallel=True)
def _adapt_tests_loop(max_before, n_discover, n_infect, n_samples, nd, p_discover, p_infect, padded_test_arr,
                      samples, test_arr):
    rng = np.random
    for i in prange(len(test_arr)):
        pi = i + max_before
        j = 0
        c = 0
        while j != n_samples:
            c += 1

            # This first loop can be avoided: simply sample from the prenormalized(!) p_discover * padded_test_arr.
            for day_infector_got_infecfted in rng.permutation(np.arange(pi - max_before + 1, pi + 1)):
                # This 0.2 is needed, because otherwise we may never discover the person on the correct day.
                # 0.2 is neglible compared to the number of tests, so shouldn't matter too much.
                this_p_discover = 0.2 + p_discover * padded_test_arr[
                                                       day_infector_got_infecfted:day_infector_got_infecfted + nd]
                s = np.sum(this_p_discover)
                this_p_discover /= s

                day_infector_discovered = day_infector_got_infecfted + rand_choice_nb(n_discover, p=this_p_discover)

                if day_infector_discovered == pi:
                    day_new_infected = day_infector_got_infecfted + rand_choice_nb(n_infect, p=p_infect)

                    this_p_discover = 1e-10 + p_discover * padded_test_arr[
                                                           day_new_infected:day_new_infected + nd]
                    s = np.sum(this_p_discover)
                    this_p_discover /= s
                    day_new_discovered = day_new_infected + rand_choice_nb(n_discover, p=this_p_discover)

                    samples[i, j] = day_new_discovered - day_infector_discovered
                    j += 1
                    break


def adapt_to_tests(ker, ker_shift, case_arr, test_arr):
    orig_ker = ker

    if test_arr is None or args.do_not_adapt:
        total_ker = ker.expand(case_arr.shape[1], -1)
        ker = ker.expand(case_arr.shape[0], case_arr.shape[1], -1)
        return ker, total_ker

    n = len(ker)
    dtype = test_arr.dtype
    device = test_arr.device

    padded_test_arr = torch.cat((test_arr[:, 0][:, None] * torch.ones((test_arr.shape[0], n), dtype=dtype, device=device),
                                 test_arr,
                                 test_arr[:, -1][:, None] * torch.ones((test_arr.shape[0], n), dtype=dtype, device=device)),
                                dim=1)

    new_ker = []
    for i, shift in enumerate(range(ker_shift, ker_shift + len(ker))):
        nk = ker[i] * torch.roll(padded_test_arr, -shift, dims=(1, ))[:, n:-n]
        new_ker.append(nk[:, :, None])

    ker = torch.cat(new_ker, dim=2) + 1e-10

    # Calculate total ker:
    total_ker = torch.sum(ker, dim=0) + 1e-10

    # Sum to 1:
    s_ker = ker.sum(dim=2)
    ker = ker / s_ker[:, :, None]
    # Fix that some communes have periods where they don't test at all:
    idxs = torch.where(s_ker == 0)
    for i in range(len(idxs[0])):
        ker[idxs[0][i], idxs[1][i], :] = orig_ker

    total_ker = total_ker / total_ker.sum(dim=1)[:, None]

    return ker, total_ker


def main():
    with open(args.simulation_file, 'rb') as f:
        data, r0_sim, k_sim, cross_sim = pickle.load(f)

    communes = [str(i) for i in range(len(data))]
    case_arr = np.array(np.round(np.array(data, dtype=np.double)[:, :-2]), dtype=int)
    population = np.ones(len(communes))

    test_arr = None  # no test data for simulations
    print(f'Running on simulated data with r0 = {np.mean(r0_sim)}, k = {k_sim}, cross = {cross_sim}')

    # Move to torch
    dtype = torch.double
    device = 'cpu' if args.cpu else 'cuda:0'   # cpu or cuda

    case_arr_long = torch.tensor(case_arr, dtype=torch.long, device=device)
    case_arr_dtype = torch.tensor(case_arr, dtype=dtype, device=device)
    if test_arr is None:
        test_arr_dtype = None
    else:
        test_arr_dtype = torch.tensor(test_arr, dtype=dtype, device=device)
    cross_base = torch.tensor(args.cross, dtype=dtype, device=device)
    norm_population = torch.tensor(population, dtype=dtype, device=device) / np.sum(population)

    # The probabilities to find at different days
    ker, p_infect, p_discover, ker_shift = likelihood_kernel(dtype, device)

    # time series should be prepended with `len(ker)` zeros (where probability is purely `cross`).
    case_arr_dtype = torch.cat((torch.zeros((case_arr_dtype.shape[0], len(ker)), dtype=dtype, device=device),
                                case_arr_dtype), dim=1)
    if test_arr_dtype is not None:
        test_arr_dtype = torch.cat((torch.ones((case_arr_dtype.shape[0], len(ker)), dtype=dtype, device=device),
                                    test_arr_dtype), dim=1)

    # Adapt ker with number of tests
    ker, total_ker = adapt_to_tests(ker, ker_shift, case_arr_dtype, test_arr_dtype)

    def prior_log_likelihood(r0, k):
        # Priors:
        beta = 5
        if k > 0:
            ll = -1 / (beta * k) - torch.log(beta * k**2)
        else:
            ll = -1e10

        if torch.any(r0 < 0.01) or torch.any(r0 > 5):
            ll += -1e10

        if args.r0_samples > 0 and args.r0_commune:
            r0 = r0[args.r0_samples:]
            # Gaussian prior
            ll = ll - torch.sum( (r0 - 1)**2 / (2 * 0.1**2) )

        return ll

    def calc_total_log_likelihood(r0, k, start_index=args.start_index,
                                  end_index=args.end_index, local_case_arr_dtype=case_arr_dtype):
        ll = prior_log_likelihood(r0, k)

        r0_scalar = True
        try:
            if len(r0) == case_arr_long.shape[0]:
                r0_scalar = False
        except TypeError:
            pass

        total = torch.sum(local_case_arr_dtype, dim=0)

        for i in range(case_arr.shape[0]):
            fraction = total * norm_population[i]
            if args.simulation:
                fraction = torch.ones_like(fraction)

            ll += log_likelihood(case_arr_long[i, :], local_case_arr_dtype[i, :],
                                 r0=r0 if r0_scalar else r0[i], k=k, cross=cross_base, total_fraction=fraction,
                                 ker=ker[i, :, :], ker_shift=ker_shift, dtype=dtype, device=device,
                                 start_index=start_index, end_index=end_index, simulation=args.simulation)
        return ll

    def optimize(f, r0_size=None):
        @lru_cache()
        def np_call(pars):
            pars = np.exp(pars)  # to force only positive values
            pars_orig = pars

            r0, k = stt(pars[0], requires_grad=True), stt(pars[1], requires_grad=True)

            local_case_arr_dtype = case_arr_dtype

            ll = f(r0, k, None, local_case_arr_dtype=local_case_arr_dtype)
            print(pars_orig, float(ll))
            logp = np.array(float(ll))
            ll.backward()
            r0_grad = r0.grad.cpu().numpy()
            k_grad = k.grad.cpu().numpy()
            grad = pars * np.hstack([r0_grad, k_grad])   # jacobian of exp transform is just pars
            return logp, grad

        def val(pars):
            return -np_call(tuple(pars))[0]

        def grad(pars):
            return -np_call(tuple(pars))[1]

        p0 = [np.log(1.0)] * (2 if r0_size is None else (1 + r0_size))
        p0[-1] = np.log(1.0)

        print('Optimization started')
        sol = scipy.optimize.minimize(val, p0, jac=grad, options={'disp': True}, tol=0.0001)
        x = np.exp(sol.x)
        return x, -sol.fun

    def stt(x, requires_grad=False):
        return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)

    def save(r0=None, k=None, cross=None, logp=None, traj_samples=None):
        if args.save_as:
            data = {'r0': r0, 'k': k, 'cross': cross, 'logp': logp, 'traj_samples':traj_samples, 'args': vars(args)}
            print('Saving', data)
            data['communes'] = communes
            data['population'] = population

            with open(args.save_as, 'wb')  as f:
                pickle.dump(data, f)
            exit()

    sampler = args.sampler  # one of 'optimize',  'hmc', 'metropolis'
    if sampler == 'optimize':
        res, logp = optimize(calc_total_log_likelihood)
    r0 = res[0]
    k = res[1]

    print('Optimum (r0, k) =', res)

    save(r0, k, logp)


if __name__ == '__main__':
    main()
