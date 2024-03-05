import numpy as np


def discretize(x, n_points):
    edges = np.linspace(x.min(), x.max(), num=n_points)
    return np.sum(x.reshape(-1, 1) > edges, axis=1)


def get_3d_1d_obs(df):
    frac_change = np.array((df.Close - df.Open) / df.Open)
    frac_high = np.array((df.High - df.Open) / df.Open)
    frac_low = np.array((df.Open - df.Low) / df.Open)

    x = discretize(frac_change, n_points=51)
    y = discretize(frac_high, n_points=11)
    z = discretize(frac_low, n_points=11)

    n = (z - 1) * (x.max() * y.max()) + (y - 1) * x.max() + x

    return n, (x, y, z)
