import pandas as pd
from pathlib import Path


def clean_index_returns(input_file, output_folder):
    input_file = Path(input_file)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_file)

    print("Loaded data:")
    print(df.head())
    print("\nShape before cleaning:", df.shape)

    # clean datetime column
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce", utc=True).dt.tz_localize(None)

    # clean numeric column
    df["y"] = pd.to_numeric(df["y"], errors="coerce")

    # drop rows where key fields are missing
    df = df.dropna(subset=["ds", "y", "series_id", "unique_id", "index_name"])

    # keep only return series
    returns_df = df[df["series_id"] == "return"].copy()

    returns_df = returns_df.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    # save cleaned return-only dataset
    output_file = output_folder / "all_indices_cleaned.csv"
    returns_df.to_csv(output_file, index=False)

    for index_name, group in returns_df.groupby("index_name"):
        out_file = output_folder / f"{index_name}_returns_cleaned.csv"
        group.to_csv(out_file, index=False)
        print(f"Saved: {out_file}")

    print("\nCleaning complete.")
    print(f"Saved cleaned returns dataset to: {output_file}")

    print("\nCleaned head:")
    print(returns_df.head())

    print("\nCleaned shape:")
    print(returns_df.shape)

    print("\nMissing values after cleaning:")
    print(returns_df.isna().sum())

    print("\nSeries counts:")
    print(returns_df["unique_id"].unique())
    print("Number of return series:", returns_df["unique_id"].nunique())

    print("\nObservations per index:")
    print(returns_df.groupby("unique_id").size())

    print("\nDate ranges per index:")
    print(returns_df.groupby("unique_id")["ds"].agg(["min", "max", "count"]))

    print("\nReturn summary:")
    print(returns_df["y"].describe())


if __name__ == "__main__":
    input_file = "/Users/melanie/Desktop/final year project/IndexData/all_indices_converted.csv"
    output_folder = "/Users/melanie/Desktop/final year project/IndexData"

    clean_index_returns(input_file=input_file, output_folder=output_folder)