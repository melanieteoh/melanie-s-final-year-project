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

USE_EXOGENOUS = False

INPUT_SIZE = 63 # roughly 3 months of trading days
MAX_STEPS = 300
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
VAL_CHECK_STEPS = 50
EARLY_STOP_PATIENCE_STEPS = 5
RANDOM_SEED = 42

# engineered feature columns
FEATURE_COLS = [
    "lag1",
    "roll_mean_5", "roll_std_5",
    "roll_mean_21", "roll_std_21",
    "dow",
]

# build lag and rolling features from historical returns only
# look-ahead leakage prevented by using .shift(1) 
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("ds").copy()

    # 1-day lag
    df["lag1"] = df["y"].shift(1) # yesterday's return

    # short-window rolling features 
    # 5 trading days is 1 week
    df["roll_mean_5"] = df["y"].shift(1).rolling(5).mean()
    df["roll_std_5"] = df["y"].shift(1).rolling(5).std()

    # long-window rolling features 
    # 21 trading days is 1 month
    df["roll_mean_21"] = df["y"].shift(1).rolling(21).mean()
    df["roll_std_21"] = df["y"].shift(1).rolling(21).std()

    # day-of-week feature
    # 0 - Monday, 1 - Tuesday, 2 - Wednesday, 3 - Thursday, 4 - Friday
    df["dow"] = df["ds"].dt.dayofweek

    return df

# load train, validation and test files 
def load_split(unique_id: str):
    train = pd.read_csv(SPLITS_DIR / f"{unique_id}_train.csv")
    val = pd.read_csv(SPLITS_DIR / f"{unique_id}_val.csv")
    test = pd.read_csv(SPLITS_DIR / f"{unique_id}_test.csv")

    for part in [train, val, test]:
        part["ds"] = pd.to_datetime(part["ds"])
        part["y"] = pd.to_numeric(part["y"], errors="coerce")

    return train, val, test

# missing dates (market holidays) are filled with y = 0.0 instead of forward-filling previous return 
def make_fedformer_ready(df, unique_id: str, use_exogenous: bool) -> pd.DataFrame:
    # remove duplicates if any 
    out = df.sort_values("ds").drop_duplicates(subset=["ds"]).copy()
    # set business-day frequency as 8
    out = out.set_index("ds").asfreq("B")
    out["y"] = out["y"].fillna(0.0)  # closed market = 0.0 return, not repeated return

    if use_exogenous:
        # forward-fill feature NaNs introduced by reindexing
        out[FEATURE_COLS] = out[FEATURE_COLS].ffill()
        out = out.reset_index()
        out["unique_id"] = unique_id
        out = out[["unique_id", "ds", "y"] + FEATURE_COLS]
    else:
        out = out.reset_index()
        out["unique_id"] = unique_id
        out = out[["unique_id", "ds", "y"]]

    return out

def build_fedformer_model(use_exogenous: bool) -> FEDformer:
    return FEDformer(
        h=HORIZON,
        input_size=INPUT_SIZE,
        hist_exog_list=FEATURE_COLS if use_exogenous else None,
        loss=MAE(), 
        learning_rate=LEARNING_RATE,
        max_steps=MAX_STEPS,
        batch_size=BATCH_SIZE,
        val_check_steps=VAL_CHECK_STEPS,
        early_stop_patience_steps=EARLY_STOP_PATIENCE_STEPS,
        scaler_type="identity",
        random_seed=RANDOM_SEED,
        accelerator="cpu",
        devices=1,
    )

# training FEDformer on train and val
# generate rolling 1 step ahead predictions over test set 
def run_fedformer_for_index(unique_id):
    train, val, test = load_split(unique_id)
    # combine train and val
    # held out test
    train_val = pd.concat([train, val], ignore_index=True).sort_values("ds")

    # build on full series ONCE before splitting or looping to avoid recomputation
    full = pd.concat([train_val, test], ignore_index=True).sort_values("ds").reset_index(drop=True)
    full = add_features(full)

    # prepare train and val slice for model fitting
    train_val_ready = make_fedformer_ready(
        full[full["ds"] < test["ds"].min()], unique_id, USE_EXOGENOUS
    ).dropna()
    print(f"Train+val rows after prep: {len(train_val_ready)}")

    # fit the model 
    model = build_fedformer_model(USE_EXOGENOUS)
    nf = NeuralForecast(models=[model], freq="B")
    # val_size tells NeuralForecast how many rows to use  for internal early-stopping validation
    nf.fit(df=train_val_ready, val_size=len(val), verbose=False)

    # prepare full series 
    # for history slicing in rolling loop
    full_ready = make_fedformer_ready(full, unique_id, USE_EXOGENOUS).dropna()

    # rolling prediction loop
    preds = []
    fail_count = 0
    nan_count = 0
    test_dates = sorted(test["ds"].tolist())

    for forecast_date in test_dates:
        # slice history strictly before forecast date 
        # prevent leakage
        history_ready = full_ready[full_ready["ds"] < forecast_date].copy()

        # FEDformer needs at least input_size rows to form a window
        if len(history_ready) < INPUT_SIZE:
            fail_count += 1
            continue 

        # only pass the last INPUT_SIZE rows
        history_slice = history_ready.tail(INPUT_SIZE).copy()

        try:
            fcst = nf.predict(df=history_ready, verbose=False)

            # NeuralForecast names prediction column after model class
            pred_col = [c for c in fcst.columns if c not in ["unique_id", "ds"]][0]

            # verify date alignment
            predicted_date = pd.Timestamp(fcst.iloc[0]["ds"])
            if predicted_date != pd.Timestamp(forecast_date):
                print(f"Date mismatch at {forecast_date}: model predicted for {predicted_date}")
                fail_count += 1
                continue

            pred_value = float(fcst.iloc[0][pred_col])
        
        except Exception as e:
            # log exact error so patterns of failure are visible
            print(f"Warning: failed at {forecast_date}: {e}")
            fail_count += 1
            continue

        # catch NaN predictions 
        if np.isnan(pred_value):
            nan_count += 1
            if nan_count <= 5:
                print(f"  NaN prediction at {forecast_date} — check model convergence")
            fail_count += 1
            continue

        # look up true return and last observed return for baselines
        actual_value = float(test[test["ds"] == forecast_date]["y"].iloc[0])
        last_return = float(full[full["ds"] < forecast_date]["y"].iloc[-1])

        preds.append({
            "unique_id": unique_id,
            "ds": forecast_date,
            "y_true": actual_value,
            "y_pred_fedformer": pred_value,
            "y_pred_zero": 0.0, # ZeroBaseline always predicts 0
            "y_pred_last": last_return, # LastReturn repeats yesterday
        })
    
    print(f"  Results: {len(preds)} succeeded, {fail_count} failed "
          f"({nan_count} were NaN) out of {len(test_dates)} test dates.")

    return pd.DataFrame(preds)

def main():
    all_predictions = []

    for unique_id in INDICES:
        print(f"Forecasting {unique_id}...")

        pred_df = run_fedformer_for_index(unique_id)

        # skip computation if no predictions produced
        if pred_df.empty:
            print(f"No predictions for {unique_id}.")
            continue

        pred_df.to_csv(OUTPUT_DIR / f"{unique_id}_predictions.csv", index=False)
        all_predictions.append(pred_df)
        print(f"Saved predictions for {unique_id}.")

    if all_predictions:
        pd.concat(all_predictions, ignore_index=True).to_csv(
            OUTPUT_DIR / "fedformer_predictions.csv", index=False
        )
        print(f"\nCombined predictions saved to {OUTPUT_DIR / 'fedformer_predictions.csv'}")
    else:
        print("\nNo predictions saved — all indices failed.")

    print("All FEDformer forecasts completed.")

if __name__ == "__main__":
    main()