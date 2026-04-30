import pandas as pd
from pathlib import Path

# feature engineering to add feature columns derived ONLY from historical returns 
# no look-ahead leakage
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("ds").copy()

    # lagged returns
    df["lag_1"] = df["y"].shift(1) # yesterday's return
    df["lag_2"] = df["y"].shift(2) # 2 days ago
    df["lag_5"] = df["y"].shift(5) # 1 week ago

    # rolling volatility (standard deviation of past returns) 
    df["vol_5"] = df["y"].shift(1).rolling(5, min_periods=3).std()
    df["vol_21"] = df["y"].shift(1).rolling(21, min_periods=10).std()

    # rolling momentum (mean of past returns) 
    df["mom_5"] = df["y"].shift(1).rolling(5, min_periods=3).mean()
    df["mom_21"] = df["y"].shift(1).rolling(21, min_periods=10).mean()

    # drop rows where features couldn't be computed (start of series)
    df = df.dropna(subset=["lag_1", "vol_5", "mom_5"])

    return df

# temporal split with strict order and no shuffling
# features built before splitting so rolling windows not broken at split boundaries
def split_one_series(df, train_end="2019-01-01", val_end="2021-01-01"):
    df = df.sort_values("ds").copy()

    train = df[df["ds"] < train_end].copy()
    val = df[(df["ds"] >= train_end) & (df["ds"] < val_end)].copy()
    test = df[df["ds"] >= val_end].copy()

    return train, val, test

def main():
    input_file = Path("/Users/melanie/Desktop/final year project/IndexData/all_indices_cleaned.csv")
    output_folder = Path("/Users/melanie/Desktop/final year project/IndexData/splits")
    output_folder.mkdir(parents=True, exist_ok=True)

    returns_df = pd.read_csv(input_file)
    returns_df["ds"] = pd.to_datetime(returns_df["ds"])

    feature_cols = ["lag_1", "lag_2", "lag_5",
                    "vol_5", "vol_21",
                    "mom_5", "mom_21"]

    print("Series found:")
    print(returns_df["unique_id"].unique())

    for unique_id in returns_df["unique_id"].unique():
        df = returns_df[returns_df["unique_id"] == unique_id].copy()

        # engineer features on full series
        df = engineer_features(df)

        # SPLIT to preserve rolling window context at boundaries 
        train, val, test = split_one_series(df)

        safe_name = unique_id.replace("/", "_")

        train.to_csv(output_folder / f"{safe_name}_train.csv", index=False)
        val.to_csv(output_folder / f"{safe_name}_val.csv", index=False)
        test.to_csv(output_folder / f"{safe_name}_test.csv", index=False)

        print(f"\n{'='*50}")
        print(f"Series: {unique_id}")
        print(f"Train: {train.shape[0]:>5} rows  ({train['ds'].min().date()} → {train['ds'].max().date()})")
        print(f"Val: {val.shape[0]:>5} rows  ({val['ds'].min().date()} → {val['ds'].max().date()})")
        print(f"Test: {test.shape[0]:>5} rows  ({test['ds'].min().date()} → {test['ds'].max().date()})")
        print(f"Features: {feature_cols}")

if __name__ == "__main__":
    main()