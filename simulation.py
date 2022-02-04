import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import scipy.special


def gamma_dist(x, r0, k):
    return np.exp(-(k * x) / r0) * (r0/k)**(-k) * x**(k - 1) / scipy.special.gamma(k)


def sim(r0, k, cross, T=100, start=1):
    x = np.arange(15)
    p = gamma_dist(x, 5, 10)
    p[p < 0] = 0.0
    p /= p.sum()

    rng = np.random.default_rng()

    people = list(rng.gamma(k, r0[0]/k, size=start))
    people_t = list(rng.integers(-len(p) - 1, 0, size=start))

    all_times = list(people_t)

    for t in range(T):
        # First clean up people who can no longer infect
        for i in reversed(list(range(len(people)))):
            if people_t[i] + len(p) <= t:
                del people_t[i]
                del people[i]

        # Infect others
        for i in range(len(people)):
            nu = people[i]
            t_i = t - people_t[i]

            n = rng.poisson(nu * p[t_i])

            people.extend(rng.gamma(k, r0[t]/k, size=n))
            people_t.extend([t] * n)
            all_times.extend([t] * n)

        for _ in range(rng.poisson(cross)):
            people.extend([rng.gamma(k, r0[t]/k)])
            people_t.extend([t])
            all_times.extend([t])

    all_times = np.array(all_times)
    all_times = all_times[all_times >= 0]

    # ### This is discover part: use a test-scenario as well rng.choice( prop to tests ).ient
    x = np.arange(15)
    p_discover = gamma_dist(x, 4.5, 5)
    p_discover /= p_discover.sum()
    discover_shift = rng.choice(x, p=p_discover, size=len(all_times))
    all_times += discover_shift
    rand_days = len(x)

    all_times_int = np.array(all_times, dtype=int)

    t, c = np.unique(all_times_int, return_counts=True)
    found = np.zeros(T + rand_days, dtype=int)
    found[t] = c

    return found[:(-rand_days if rand_days > 0 else None)]


def simplesim(r0, k, cross, T=100):
    rng = np.random.default_rng()

    n = 3
    found = [n]
    for t in range(T):
        nu = rng.gamma(k, r0 / k, size=n + (1 if rng.random() <= cross else 0))
        n = rng.poisson(nu).sum()
        found.append(n)
    return found


def make_r0_profile(r0, T, p1=0.05, p2=0.25, min_t=15):
    last_t = -10000
    r = np.ones(T)
    for t in range(T):
        if np.random.random() < p1 and t - last_t > min_t:
            r[t:] *= 1 - p2 + 2 * p2 * np.random.random()
            last_t = t

    r = r / np.mean(r) * r0

    return r


def main():
    r0 = 1.1
    k = 0.1
    cross = 0.05
    T = 150
    skip = 5
    end = 10
    start = 1
    n_communes = 250
    r0 = r0 * np.ones(T + skip + end)

    data = []
    for _ in tqdm(range(n_communes)):
        c = sim(r0, k, cross, T + skip + end, start)[skip:-end]

        if np.sum(c) > 0:
            data.append(c)
            plt.plot(c)

    with open('simulation.pkl', 'wb') as f:
        pickle.dump((data, r0, k, cross), f)


if __name__ == '__main__':
    main()
