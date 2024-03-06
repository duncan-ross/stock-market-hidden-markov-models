from hmmlearn.hmm import GMMHMM
import numpy as np
from tqdm import tqdm


def discretize(x, n_points, edges=None):
    if edges is None:
        edges = np.linspace(x.min(), x.max(), num=n_points)
    return np.sum(x.reshape(-1, 1) > edges, axis=1), edges


def get_3d_1d_obs(df, edges=None, x_max=None, y_max=None):
    frac_change = np.array((df.Close - df.Open) / df.Open)
    frac_high = np.array((df.High - df.Open) / df.Open)
    frac_low = np.array((df.Open - df.Low) / df.Open)

    if edges is None:
        change_edges, high_edges, low_edges = None, None, None
    else:
        change_edges, high_edges, low_edges = edges
    x, change_edges = discretize(frac_change, n_points=51, edges=change_edges)
    y, high_edges = discretize(frac_high, n_points=11, edges=high_edges)
    z, low_edges = discretize(frac_low, n_points=11, edges=low_edges)

    if x_max is None:
        x_max = x.max()
    if y_max is None:
        y_max = y.max()

    n = (z - 1) * (x.max() * y.max()) + (y - 1) * x.max() + x

    return (
        n,
        (x, y, z),
        (frac_change, frac_high, frac_low),
        (change_edges, high_edges, low_edges),
    )


def get_hmm_input(n):
    n = n.reshape(-1, 1)
    for i in range(1, 10):
        n = np.hstack((n, np.roll(n[:, 0], -i).reshape(-1, 1)))
    n = n[:-9]
    return n.reshape(-1, 1)


def get_data_subset(df, start_date, end_date, include_buffer=True):
    df.reset_index(drop=True, inplace=True)
    start_idx = np.arange(df.shape[0])[df.Date == start_date].item()
    if include_buffer:
        start_idx -= 9
    end_idx = np.arange(df.shape[0])[df.Date == end_date].item()
    return df.iloc[start_idx : (end_idx + 1)]


def train_hmm(df, start_date, end_date, include_buffer=True):
    train_df = get_data_subset(df, start_date, end_date, include_buffer)
    n, (x, y, _), fracs, edges = get_3d_1d_obs(train_df)
    x_hmm = get_hmm_input(n)
    hmm_model = GMMHMM(
        n_components=4, covariance_type="diag", random_state=217, n_iter=1000
    )
    hmm_model.fit(X=x_hmm, lengths=[10 for _ in range(x_hmm.shape[0] // 10)])
    return hmm_model, (x.max(), y.max()), fracs, edges


def predict_hmm(
    df,
    start_date,
    end_date,
    hmm_model,
    frac_change,
    edges,
    x_max,
    y_max,
    include_buffer=True,
):
    test_df = get_data_subset(df, start_date, end_date, include_buffer)
    n, _, _, _ = get_3d_1d_obs(test_df, edges, x_max, y_max)
    x_hmm_test = get_hmm_input(n).reshape(-1, 10)[:, :9]
    best_xs = []
    print("Running Predictions")
    for x in tqdm(x_hmm_test):
        best_ll = float("-inf")
        for x in range(50):
            for y in range(10):
                for z in range(10):
                    n_i = (z - 1) * (x_max * y_max) + (y - 1) * x_max + x
                    score = hmm_model.score(
                        np.concatenate((x, np.array([n_i]))).reshape(-1, 1)
                    )
                    if score > best_ll:
                        best_x = x
                        best_ll = score
        best_xs.append(best_x)

    if include_buffer:
        test_df = test_df.iloc[9:]
    s = test_df.Open.to_numpy()
    c = test_df.Close.to_numpy()
    p = (1 + np.array([frac_change[x] for x in best_xs])) * s

    return s, c, p


def calculate_mape_dpa(p, c, s):
    mape = np.mean(np.abs(p - c) / c)
    dpa = np.mean(np.sign(p - s) == np.sign(c - s))
    return mape, dpa
