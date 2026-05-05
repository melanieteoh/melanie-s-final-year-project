import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path

from neuralforecast import NeuralForecast
from neuralforecast.models import FEDformer
from neuralforecast.losses.pytorch import MAE

logging.basicConfig(level=logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("neuralforecast").setLevel(logging.ERROR)

PROJECT_ROOT = Path("/Users/melanie/Desktop/final year project")

SPLITS_DIR = PROJECT_ROOT / "IndexData" / "splits"

OUTPUT_DIR = PROJECT_ROOT / "Outputs" / "fedformer"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INDICES = [
    "RUSSELL1000_return",
    "TOPIX1000_return",
]

# model hyperparameters
HORIZON = 1 # 1 to predict next day's return
INPUT_SIZE = 63 # roughly 3 months of trading days
MAX_STEPS = 300 # max gradient steps during training
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
VAL_CHECK_STEPS = 50 # how often to check val loss for early stopping
EARLY_STOP_PATIENCE_STEPS = 5  # stop after 5 bad val checks in a row
RANDOM_SEED = 42

# load train, validation and test files 
def load_split(unique_id: str):
    train = pd.read_csv(SPLITS_DIR / f"{unique_id}_train.csv")
    val = pd.read_csv(SPLITS_DIR / f"{unique_id}_val.csv")
    test = pd.read_csv(SPLITS_DIR / f"{unique_id}_test.csv")

    for part in [train, val, test]:
        part["ds"] = pd.to_datetime(part["ds"])
        part["y"] = pd.to_numeric(part["y"], errors="coerce")

    return train, val, test

# missing dates filled with y = 0.0 instead of forward-filling
def make_fedformer_ready(hist_df: pd.DataFrame, unique_id: str) -> pd.DataFrame:
    # remove duplicates if any
    hist_df = hist_df.sort_values("ds").drop_duplicates(subset=["ds"]).copy()

    # set business-day frequency
    hist_df = hist_df.set_index("ds").asfreq("B")

    # closed market = 0.0 return, not repeated return
    hist_df["y"] = hist_df["y"].fillna(0.0)

    hist_df = hist_df.reset_index()
    hist_df["unique_id"] = unique_id

    return hist_df[["unique_id", "ds", "y"]]

# FEDformer model with the chosen hyperparameters
def build_fedformer_model() -> FEDformer:
    return FEDformer(
        h=HORIZON,
        input_size=INPUT_SIZE,
        loss=MAE(),
        learning_rate=LEARNING_RATE,
        max_steps=MAX_STEPS,
        batch_size=BATCH_SIZE,
        val_check_steps=VAL_CHECK_STEPS,
        early_stop_patience_steps=EARLY_STOP_PATIENCE_STEPS,
        scaler_type="standard",
        random_seed=RANDOM_SEED,
        accelerator="cpu",
        devices=1,
    )

# FEDformer rolling forecast function
def run_fedformer_for_index(unique_id: str) -> pd.DataFrame:
    train, val, test = load_split(unique_id)

    # initial history available before testing starts
    history = pd.concat([train, val], ignore_index=True)
    history = history.sort_values("ds").reset_index(drop=True)

    test = test.sort_values("ds").reset_index(drop=True)

    # trained locally
    # train once on train+val, then reuse the same trained model for every rolling step
    train_ready = make_fedformer_ready(history, unique_id)

    print(f"{unique_id}: training FEDformer on {len(train_ready)} rows...", flush=True)

    model = build_fedformer_model()
    nf = NeuralForecast(models=[model], freq="B")
    # val_size is number of trailing rows reserved for internal early-stopping
    nf.fit(df=train_ready, val_size=len(val), verbose=False)
    print(f"{unique_id}: training complete.", flush=True)

    preds = []
    fail_count = 0

    for i in range(len(test)):
        forecast_row = test.iloc[[i]].copy()
        forecast_date = forecast_row["ds"].iloc[0]

        print(
            f"{unique_id}: forecasting {i + 1}/{len(test)} for {forecast_date.date()}...",
            flush=True,
        )

        # build the history the model will see (everything up to t-1)
        history_ready = make_fedformer_ready(history, unique_id)

        # FEDformer needs at least input_size rows to form a window
        # this should never trigger on real splits but helps keep script safe
        if len(history_ready) < INPUT_SIZE:
            print(f"  Not enough history ({len(history_ready)} < {INPUT_SIZE}), skipping")
            fail_count += 1
            # still reveal the actual test row to keep the rolling history aligned
            history = pd.concat([history, forecast_row], ignore_index=True)
            history = history.sort_values("ds").reset_index(drop=True)
            continue

        try:
            # NeuralForecast predicts h steps after last date in df
            forecast = nf.predict(df=history_ready, verbose=False)

            # NeuralForecast names the prediction column after the model class
            pred_col = [c for c in forecast.columns if c not in ["unique_id", "ds"]][0]
            predicted_date = pd.Timestamp(forecast.iloc[0]["ds"])
            pred_value = float(forecast.iloc[0][pred_col])

            # verify date alignment 
            # model's predicted date should match  next test date because both run on business-day frequency
            if predicted_date != pd.Timestamp(forecast_date):
                print(
                    f"Date mismatch: model predicted {predicted_date.date()} "
                    f"but expected {forecast_date.date()}. Skipping."
                )
                fail_count += 1
                history = pd.concat([history, forecast_row], ignore_index=True)
                history = history.sort_values("ds").reset_index(drop=True)
                continue

            # catch NaN predictions
            # happens if training under-converges
            if np.isnan(pred_value):
                print(f"  NaN prediction at {forecast_date.date()}")
                fail_count += 1
                history = pd.concat([history, forecast_row], ignore_index=True)
                history = history.sort_values("ds").reset_index(drop=True)
                continue

            print(
                f"{unique_id}: done {i + 1}/{len(test)} for {forecast_date.date()} "
                f"(pred={pred_value:.6f})",
                flush=True,
            )

        except Exception as e:
            # log exact error so patterns of failure are visible
            print(f"Warning: prediction failed at {forecast_date.date()}: {e}")
            fail_count += 1
            history = pd.concat([history, forecast_row], ignore_index=True)
            history = history.sort_values("ds").reset_index(drop=True)
            continue

        actual_value = float(forecast_row["y"].iloc[0])
        last_return_baseline = float(history["y"].iloc[-1])

        preds.append({
            "unique_id": unique_id,
            "ds": forecast_date,
            "y_true": actual_value,
            "y_pred_fedformer": pred_value,
            "y_pred_zero": 0.0, # ZeroBaseline always predicts 0
            "y_pred_last": last_return_baseline # LastReturn repeats yesterday's return
        })

        # reveal the actual test row and append it to history (rolling step)
        history = pd.concat([history, forecast_row], ignore_index=True)
        history = history.sort_values("ds").reset_index(drop=True)

    print(
        f"Results: {len(preds)} succeeded, {fail_count} failed "
        f"out of {len(test)} test dates."
    )
    return pd.DataFrame(preds)

def main():
    all_predictions = []

    for unique_id in INDICES:
        print(f"Forecasting {unique_id}...")

        pred_df = run_fedformer_for_index(unique_id)

        if pred_df.empty:
            print(f"No predictions produced for {unique_id}")
            continue

        # save per-index predictions
        pred_df.to_csv(OUTPUT_DIR / f"{unique_id}_predictions.csv", index=False)
        all_predictions.append(pred_df)
        print(f"  Done with {unique_id}.")

    if all_predictions:
        pd.concat(all_predictions, ignore_index=True).to_csv(
            OUTPUT_DIR / "fedformer_predictions.csv",
            index=False,
        )
        print(f"\nCombined predictions saved to {OUTPUT_DIR / 'fedformer_predictions.csv'}")
    else:
        print("\nNo predictions saved — all indices failed.")

    print("All FEDformer forecasts completed.")

if __name__ == "__main__":
    main()