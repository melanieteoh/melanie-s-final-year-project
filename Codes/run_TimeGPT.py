import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from nixtla import NixtlaClient

logging.basicConfig(level=logging.ERROR)
logging.getLogger("nixtla").setLevel(logging.ERROR)
logging.getLogger("nixtla.nixtla_client").setLevel(logging.ERROR)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPLITS_DIR = PROJECT_ROOT / "IndexData" / "splits"

OUTPUT_DIR = PROJECT_ROOT / "Outputs" / "timegpt"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# indices used
INDICES = [
    "RUSSELL1000_return",
    "TOPIX1000_return",
]

# model hyperparameters
HORIZON = 1  # 1 to predict next day's return

API_KEY = os.getenv("NIXTLA_API_KEY")

# load train, validation and test files
# TimeGPT is used univariately so only ds and y are needed
def load_split(unique_id: str):
    train = pd.read_csv(SPLITS_DIR / f"{unique_id}_train.csv")
    val   = pd.read_csv(SPLITS_DIR / f"{unique_id}_val.csv")
    test  = pd.read_csv(SPLITS_DIR / f"{unique_id}_test.csv")

    for part in [train, val, test]:
        part["ds"] = pd.to_datetime(part["ds"])
        part["y"]  = pd.to_numeric(part["y"], errors="coerce")

    return train, val, test

# missing dates (market holidays) are filled with y = 0.0 instead of forward-filling previous return
def make_timegpt_ready(hist_df: pd.DataFrame, unique_id: str) -> pd.DataFrame:
    # remove duplicates if any
    hist_df = hist_df.sort_values("ds").drop_duplicates(subset=["ds"]).copy()
    # set business-day frequency
    hist_df = hist_df.set_index("ds").asfreq("B")

    # closed market = 0.0 return, not repeated return
    hist_df["y"] = hist_df["y"].fillna(0.0)

    hist_df = hist_df.reset_index()
    hist_df["unique_id"] = unique_id

    return hist_df

# timeGPT rolling forecast function
def run_timegpt_for_index(unique_id: str, client: NixtlaClient) -> pd.DataFrame:
    train, val, test = load_split(unique_id)

    # initial history available before testing starts
    history = pd.concat([train, val], ignore_index=True)
    history = history.sort_values("ds").reset_index(drop=True)

    test = test.sort_values("ds").reset_index(drop=True)

    preds = []
    fail_count = 0

    for i in range(len(test)):
        forecast_row  = test.iloc[[i]].copy()
        forecast_date = forecast_row["ds"].iloc[0]

        print(
            f"{unique_id}: forecasting {i + 1}/{len(test)} for {forecast_date.date()}...",
            flush=True,
        )

        # target history for TimeGPT (univariate: only ds and y)
        history_ready = make_timegpt_ready(
            history[["ds", "y"]].copy(), unique_id
        )

        try:
            forecast = client.forecast(
                df=history_ready,
                h=HORIZON,
                freq="B",
            )

            # Nixtla names the prediction column after model variant
            pred_col   = [c for c in forecast.columns if c not in ["unique_id", "ds"]][0]
            pred_value = float(forecast.iloc[0][pred_col])

            print(
                f"{unique_id}: done {i + 1}/{len(test)} for {forecast_date.date()}",
                flush=True,
            )

        except Exception as e:
            # log exact error so patterns of failure are visible
            print(f"Warning: prediction failed at {forecast_date}: {e}")
            fail_count += 1
            continue

        actual_value = float(forecast_row["y"].iloc[0])
        last_return_baseline = float(history["y"].iloc[-1])

        preds.append({
            "unique_id": unique_id,
            "ds": forecast_date,
            "y_true": actual_value,
            "y_pred_timegpt": pred_value,
            "y_pred_zero": 0.0,
            "y_pred_last": last_return_baseline,
        })

        # reveal the actual test row and append it to history
        history = pd.concat([history, forecast_row], ignore_index=True)
        history = history.sort_values("ds").reset_index(drop=True)

    return pd.DataFrame(preds)

def main():
    client = NixtlaClient(api_key=API_KEY)

    all_predictions = []

    for unique_id in INDICES:
        print(f"Forecasting {unique_id}...")

        pred_df = run_timegpt_for_index(unique_id, client)

        if pred_df.empty:
            print(f"No predictions produced for {unique_id}")
            continue

        # save per-index predictions
        pred_df.to_csv(OUTPUT_DIR / f"{unique_id}_predictions.csv", index=False)
        all_predictions.append(pred_df)
        print(f"  Done with {unique_id}.")

    if all_predictions:
        pd.concat(all_predictions, ignore_index=True).to_csv(
            OUTPUT_DIR / "timegpt_predictions.csv",
            index=False,
        )

    print("All TimeGPT forecasts completed.")

if __name__ == "__main__":
    main()
