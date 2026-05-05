import logging
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor

logging.basicConfig(level=logging.ERROR)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPLITS_DIR = PROJECT_ROOT / "IndexData" / "splits"

OUTPUT_DIR = PROJECT_ROOT / "Outputs" / "xgboost"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INDICES = [
    "RUSSELL1000_return",
    "TOPIX1000_return",
]

# inputs XGBoost used to predict y
FEATURE_COLS = [
    "lag_1", "lag_2", "lag_5", # short to medium lagged returns
    "vol_5", "vol_21", # short and long volatility
    "mom_5", "mom_21" # short and long momentum
]

# XGBoost hyperparameters
N_ESTIMATORS = 200 # number of boosting rounds
MAX_DEPTH = 4 # tree depth (shallow to limit overfitting)
LEARNING_RATE = 0.05 # small step size paired with many trees
SUBSAMPLE = 0.8 # row sampling per tree for variance reduction
COLSAMPLE = 0.8 # column sampling per tree
RANDOM_SEED = 42 # fixed

def load_split(unique_id: str):
    train = pd.read_csv(SPLITS_DIR / f"{unique_id}_train.csv")
    val   = pd.read_csv(SPLITS_DIR / f"{unique_id}_val.csv")
    test  = pd.read_csv(SPLITS_DIR / f"{unique_id}_test.csv")

    for part in [train, val, test]:
        part["ds"] = pd.to_datetime(part["ds"])
        part["y"]  = pd.to_numeric(part["y"], errors="coerce")

    return train, val, test

# train one XGBoost model per index and predict the test set in one batch
def run_xgboost_for_index(unique_id: str):
    train, val, test = load_split(unique_id)

    # combine train + val for fitting 
    train_val = pd.concat([train, val], ignore_index=True)
    train_val = train_val.sort_values("ds").reset_index(drop=True)

    # drop rows with NaN features
    # vol_21 and mom_21 have NaNs in the first 21 rows because their rolling windows do not have enough history
    train_val  = train_val.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    test_clean = test.dropna(subset=FEATURE_COLS).reset_index(drop=True)

    print(
        f"{unique_id}: training XGBoost on {len(train_val)} rows, "
        f"predicting {len(test_clean)} test rows...",
        flush=True,
    )

    X_train = train_val[FEATURE_COLS]
    y_train = train_val["y"]
    X_test  = test_clean[FEATURE_COLS]
    y_test  = test_clean["y"]

    model = XGBRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE,
        objective="reg:squarederror",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # predict the entire test set in one batch
    # key difference from TimeGPT and FEDformer which use rolling 1-step ahead
    y_pred = model.predict(X_test)

    # baselines for parity with the other models' output format
    preds = pd.DataFrame({
        "unique_id": unique_id,
        "ds": test_clean["ds"].values,
        "y_true": y_test.values,
        "y_pred_xgboost": y_pred,
        "y_pred_zero": 0.0,
        "y_pred_last": test_clean["lag_1"].values,
    })

    print(f"{unique_id}: done.")
    return preds, model

def main():
    all_predictions = []

    for unique_id in INDICES:
        print(f"Forecasting {unique_id}...")

        pred_df, model = run_xgboost_for_index(unique_id)

        if pred_df.empty:
            print(f"No predictions produced for {unique_id}")
            continue

        pred_df.to_csv(OUTPUT_DIR / f"{unique_id}_predictions.csv", index=False)

        model.save_model(OUTPUT_DIR / f"{unique_id}_xgboost.json")

        all_predictions.append(pred_df)
        print(f"  Saved predictions and model for {unique_id}.")

    if all_predictions:
        pd.concat(all_predictions, ignore_index=True).to_csv(
            OUTPUT_DIR / "xgboost_predictions.csv",
            index=False,
        )
        print(f"\nCombined predictions saved to {OUTPUT_DIR / 'xgboost_predictions.csv'}")
    else:
        print("\nNo predictions saved — all indices failed.")

    print("All XGBoost forecasts completed.")

if __name__ == "__main__":
    main()
