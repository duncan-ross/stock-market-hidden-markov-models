from constants import train_trials
from functions import load_data, run_trial

if __name__ == "__main__":
    dfs = load_data(tcks=("AAPL", "DELL", "FORD", "IBM", "MACYS", "SP500"))

    # Replicate trials from the paper
    mapes = {}
    dpas = {}
    preds = {}

    # Paper constants (adjust as needed)
    latency = 10
    n_states = 4

    for tck, trials in train_trials.items():
        mapes[tck] = {}
        dpas[tck] = {}
        for i, trial in trials.items():
            # TODO: Write predictions to CSV file
            results = run_trial(
                dfs,
                tck=tck, train_period=trial["train"], test_period=trial["test"],
                latency=latency, n_states=n_states
            )
            mapes[tck][i] = results["MAPE"]
            dpas[tck][i] = results["DPA"]
