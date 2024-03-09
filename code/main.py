import pandas as pd
from constants import train_trials
from functions import load_data, run_trial

if __name__ == "__main__":
    dfs = load_data(tcks=("AAPL", "DELL", "FORD", "IBM", "MACYS", "SP500"))

    # Paper constants (adjust as needed)
    latency = [6, 10, 20, 50]
    n_states = [3, 4, 5, 6]

    for tck, trials in train_trials.items():
        preds = pd.DataFrame()
        for i, trial in trials.items():
            for lat in latency:
                for n in n_states:
                    results, preds_df = run_trial(
                        dfs,
                        tck=tck, train_period=trial["train"], test_period=trial["test"],
                        latency=lat, n_states=n
                    )
                    preds_df["experiment_id"] = i
                    preds_df["min_train_date"] = trial["train"][0]
                    preds_df["max_train_date"] = trial["train"][1]
                    preds_df["latency"] = lat
                    preds_df["n_states"] = n
                    preds = pd.concat([preds, preds_df])
        preds.to_csv(f"../predictions/{tck}/predictions.csv", index=False)
