from constants import train_trials
from functions import load_data, run_trial
import pandas as pd

if __name__ == "__main__":
    dfs = load_data(tcks=("AAPL", "DELL", "FORD", "IBM", "MACYS", "SP500"))

    # Paper constants (adjust as needed)
    additional_latency = [6, 20, 50]
    additional_n_states = [3, 5, 6, 8]

    for tck, trials in train_trials.items():
        preds = pd.DataFrame()
        for i, trial in trials.items():
            if tck == "AAPL" and i == 1:
                for lat in additional_latency:
                    results, preds_df = run_trial(
                        dfs,
                        tck=tck,
                        train_period=trial["train"],
                        test_period=trial["test"],
                        latency=lat,
                        n_states=4,
                    )
                    preds_df["experiment_id"] = i
                    preds_df["min_train_date"] = trial["train"][0]
                    preds_df["max_train_date"] = trial["train"][1]
                    preds_df["latency"] = lat
                    preds_df["n_states"] = 4
                    preds = pd.concat([preds, preds_df])
                for n in additional_n_states:
                    results, preds_df = run_trial(
                        dfs,
                        tck=tck,
                        train_period=trial["train"],
                        test_period=trial["test"],
                        latency=10,
                        n_states=n,
                    )
                    preds_df["experiment_id"] = i
                    preds_df["min_train_date"] = trial["train"][0]
                    preds_df["max_train_date"] = trial["train"][1]
                    preds_df["latency"] = 10
                    preds_df["n_states"] = n
                    preds = pd.concat([preds, preds_df])
            results, preds_df = run_trial(
                dfs,
                tck=tck,
                train_period=trial["train"],
                test_period=trial["test"],
                latency=10,
                n_states=4,
            )
            preds_df["experiment_id"] = i
            preds_df["min_train_date"] = trial["train"][0]
            preds_df["max_train_date"] = trial["train"][1]
            preds_df["latency"] = 10
            preds_df["n_states"] = 4
            preds = pd.concat([preds, preds_df])

        preds.to_csv(f"../predictions/{tck}/predictions.csv", index=False)
