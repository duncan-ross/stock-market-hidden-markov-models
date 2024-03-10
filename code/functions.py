import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from hmmlearn.hmm import GMMHMM
from tqdm import tqdm


def process_prices(df, col_mapper):
    for in_col, out_col in col_mapper.items():
        df[out_col] = [float(re.findall("(?<=\$).*", x)[0]) for x in df[in_col]]
    return df


def load_data(tcks):
    dfs = {}
    for tck in tcks:
        if tck in ("AAPL", "IBM"):
            date_format = "%Y-%m-%d"
        elif tck == "DELL":
            date_format = "%m/%d/%Y"
        else:
            date_format = "%m/%d/%y"

        df = pd.read_csv(
            f"../data/input/{tck}.csv", parse_dates=["Date"], date_format=date_format
        )
        # Only include date, not time
        df["Date"] = [re.findall("^.*(?=\s)", str(x))[0] for x in df["Date"]]

        if tck == "DELL":
            column_mapping = {
                "Close/Last": "Close",
                "Open": "Open",
                "High": "High",
                "Low": "Low",
            }
            df = process_prices(df, col_mapper=column_mapping)

        dfs[tck] = df.sort_values("Date")

    return dfs


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


def get_hmm_input(n, latency=10):
    n = n.reshape(-1, 1)
    for i in range(1, latency):
        n = np.hstack((n, np.roll(n[:, 0], -i).reshape(-1, 1)))
    n = n[: -(latency - 1)]
    return n.reshape(-1, 1)


def get_data_subset(df, start_date, end_date, latency=10, include_buffer=True):
    df.reset_index(drop=True, inplace=True)
    start_idx = np.arange(df.shape[0])[df.Date == start_date].item()
    if include_buffer:
        start_idx -= latency - 1
    end_idx = np.arange(df.shape[0])[df.Date == end_date].item()
    return df.iloc[start_idx : (end_idx + 1)]


def train_hmm(df, start_date, end_date, latency=10, n_states=4, include_buffer=True):
    train_df = get_data_subset(df, start_date, end_date, latency, include_buffer)
    n, (x, y, _), fracs, edges = get_3d_1d_obs(train_df)
    x_hmm = get_hmm_input(n, latency=latency)
    hmm_model = GMMHMM(
        n_components=n_states,
        n_mix=4,
        covariance_type="diag",
        random_state=217,
        n_iter=1000,
	    algorithm="map"
    )
    hmm_model.fit(
        X=x_hmm,
        lengths=[latency for _ in range(x_hmm.shape[0] // latency)]
    )
    return hmm_model, (x.max(), y.max()), fracs, edges


def process_x_hmm(args):
    x_hmm, x_max, y_max, hmm_model = args
    best_ll = float("-inf")
    best_x = None  # Initialize best_x
    for x in range(50):
        for y in range(10):
            for z in range(10):
                n_i = (z - 1) * (x_max * y_max) + (y - 1) * x_max + x
                score = hmm_model.score(np.concatenate((x_hmm, np.array([n_i]))).reshape(-1, 1))
                if score > best_ll:
                    best_x = x
                    best_ll = score
    return best_x

def predict_hmm(
    df,
    start_date,
    end_date,
    hmm_model,
    frac_change,
    edges,
    x_max,
    y_max,
    latency=10,
    include_buffer=True,
):
    # Assuming get_data_subset, get_3d_1d_obs, and get_hmm_input are defined elsewhere
    test_df = get_data_subset(df, start_date, end_date, latency, include_buffer)
    n, _, _, _ = get_3d_1d_obs(test_df, edges, x_max, y_max)
    x_hmm_test = get_hmm_input(n, latency=latency).reshape(-1, latency)[:, : (latency - 1)]
    
    # Prepare data for parallel processing
    tasks = [(x_hmm, x_max, y_max, hmm_model) for x_hmm in x_hmm_test]
    
    # Parallel processing
    best_xs = []
    with ProcessPoolExecutor() as executor:
        futures = list(executor.map(process_x_hmm, tasks))
        for future in tqdm(futures):
            best_xs.append(future)

    if include_buffer:
        test_df = test_df.iloc[(latency - 1) :]
    s = test_df.Open.to_numpy()
    c = test_df.Close.to_numpy()
    p = (1 + np.array([frac_change[x] for x in best_xs])) * s

    return s, c, p


def calculate_mape_dpa(p, c, s):
    mape = np.mean(np.abs(p - c) / c)
    dpa = np.mean(np.sign(p - s) == np.sign(c - s))
    return {"MAPE": mape, "DPA": dpa}


def run_trial(dfs, tck, train_period, test_period, latency=10, n_states=4):
    """
    Run a single trial on a select train/test period, for specific company,
    with specific hyperparameters

    Parameters
    ----------
    dfs: dict
        Dictionary of company stock price DataFrames
    tck: str
        Ticker or prefix for CSV files with stock price data
        e.g., "AAPL", "DELL", "FORD", "IBM", "MACYS", "SP500"
    train_period: tuple, length 2
        Length-2 tuple with the start and end dates for the training period
        e.g., ("2021-01-04", "2022-01-03")
    train_period: tuple, length 2
        Length-2 tuple with the start and end dates for the test period
        e.g., ("2023-01-03", "2023-07-11")
    latency: int
        How many timesteps to look back in the HMM
    """
    df = dfs[tck]
    start_train, end_train = train_period
    start_test, end_test = test_period

    print(f"Training HMM on train period {train_period}")
    hmm_model, (x_max, y_max), (train_fracs), train_edges = train_hmm(
        df,
        start_date=start_train,
        end_date=end_train,
        latency=latency,
        n_states=n_states,
    )
    print("Training Complete")

    print(f"Generating Predictions for test period {test_period}")
    s_test, c_test, p_test = predict_hmm(
        df,
        start_date=start_test,
        end_date=end_test,
        hmm_model=hmm_model,
        frac_change=train_fracs[0],
        edges=train_edges,
        x_max=x_max,
        y_max=y_max,
        latency=latency,
    )
    print("Predictions Generated")

    # TODO: Write predictions to CSV file
    preds_df = pd.DataFrame(
        {
            "date": df.Date[(df.Date >= start_test) & (df.Date <= end_test)],
            "open": s_test,
            "close": c_test,
            "predicted": p_test,
            "percentage_change": (c_test - s_test) / s_test,
            "predicted_percentage_change": (p_test - s_test) / s_test,
            "mape": np.abs(p_test - c_test) / c_test,
            "dpa": np.sign(p_test - s_test) == np.sign(c_test - s_test),
            "ticker": tck,
            "latency": latency,
        }
    )

    results = calculate_mape_dpa(p=p_test, c=c_test, s=s_test)
    print(
        f"{tck} trial with latent states = {n_states}, context window size = {latency}"
    )
    print(f"MAPE = {results['MAPE']}, DPA = {results['DPA']}")
    print("-" * 80)
    return results, preds_df


def load_predictions(ticker, latency: int = None, n_states: int = None):
    df = pd.read_csv(f"../predictions/{ticker}/predictions.csv")
    if latency:
        df = df[df["latency"] == latency]
    if n_states:
        df = df[df["n_states"] == n_states]
    df["date"] = pd.to_datetime(df["date"])
    return df
