import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from nixtla import NixtlaClient

logging.basicConfig(level=logging.ERROR)
logging.getLogger("nixtla").setLevel(logging.ERROR)
logging.getLogger("nixtla.nixtla_client").setLevel(logging.ERROR)

PROJECT_ROOT = Path("/Users/melanie/Desktop/final year project")

SPLITS_DIR = PROJECT_ROOT / "IndexData" / "splits"

OUTPUT_DIR = PROJECT_ROOT / "Outputs" / "timegpt"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# indices used
INDICES = [
    "RUSSELL1000_return",
    "TOPIX1000_return",
]

# model hyperparameters
HORIZON = 1 # 1 to predict next day's return

USE_EXOGENOUS = False

API_KEY = os.getenv("NIXTLA_API_KEY")

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
def make_timegpt_ready(hist_df: pd.DataFrame, unique_id: str, include_features: bool = False) -> pd.DataFrame:
    # if include_features = True, feature columns are aligned and filled
    # remove duplicates if any 
    hist_df = hist_df.sort_values("ds").drop_duplicates(subset=["ds"]).copy()
    # set business-day frequency as 8
    hist_df = hist_df.set_index("ds").asfreq("B") 

    # closed market = 0.0 return, not repeated return
    hist_df["y"] = hist_df["y"].fillna(0.0)

    # forwall-fill feature NaNs introduced by reindexing to business days 
    if include_features:
        hist_df[FEATURE_COLS] = hist_df[FEATURE_COLS].ffill()
    
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
        forecast_row = test.iloc[[i]].copy()
        forecast_date = forecast_row["ds"].iloc[0]

        print(
            f"{unique_id}: forecasting {i + 1}/{len(test)} for {forecast_date.date()}...",
            flush=True
        )

        # target history for TimeGPT
        history_ready = make_timegpt_ready(
            history[["ds", "y"]].copy(), unique_id
        )

        try:
            if USE_EXOGENOUS:
                # build feature history from current history up to t-1
                history_with_features = add_features(history.copy())
                history_exog = make_timegpt_ready(
                    history_with_features[["ds","y"] + FEATURE_COLS].copy(),
                    unique_id,
                    include_features = True,
                ).dropna(subset=FEATURE_COLS)

                # build the one-step future row correctly using history and current forecast date
                temp = pd.concat([history.copy(), forecast_row.copy()], ignore_index=True).sort_values("ds")
                temp = add_features(temp)

                futr_row = temp.iloc[[-1]][["ds"] + FEATURE_COLS].copy()
                futr_row["unique_id"] = unique_id

                # skip if features still NaN
                if futr_row[FEATURE_COLS].isna().any(axis=1).iloc[0]:
                    print(f"Skipping {forecast_date}: NaNs in future feature row")
                    fail_count += 1
                    continue

                forecast = client.forecast(
                    df=history_ready,
                    X_df=history_exog,
                    futr_df=futr_row,
                    h=HORIZON,
                    freq="B",
                )

            else:
                # forecast without exogenous variables
                forecast = client.forecast(
                    df=history_ready,
                    h=HORIZON,
                    freq="B",
                )

            # Nixtla names the prediction column after model variant
            pred_col = [c for c in forecast.columns if c not in ["unique_id", "ds"]][0]
            pred_value = float(forecast.iloc[0][pred_col])

            print(f"{unique_id}: done {i + 1}/{len(test)} for {forecast_date.date()}",flush=True)

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
            OUTPUT_DIR / "timeGPT_predictions.csv",
            index=False
        )

    print("All TimeGPT forecasts completed.")

if __name__ == "__main__":
    main()