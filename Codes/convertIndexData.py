import bz2
import json
import pandas as pd
from pathlib import Path

def json_bz2_to_long_table(file_path):
    file_path = Path(file_path)
    index_name = file_path.stem.replace(".json", "")

    with bz2.open(file_path, "rt") as f:
        data = json.load(f)

    # convert nested dictionary into a wide DF
    wide_df = pd.DataFrame.from_dict(data, orient="index")
    wide_df.index = pd.to_datetime(wide_df.index)
    wide_df.index.name = "ds"

    # convert wide to long
    long_df = (
        wide_df
        .stack(dropna=False)
        .reset_index()
        .rename(columns={"level_1": "series_id", 0: "y"})
    )

    long_df["index_name"] = index_name
    long_df["series_id"] = long_df["series_id"].astype(str)
    long_df["unique_id"] = long_df["index_name"] + "_" + long_df["series_id"]

    long_df = long_df[["index_name", "series_id", "unique_id", "ds", "y"]]
    long_df = long_df.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    return long_df

folder = Path("/Users/melanie/Desktop/final year project/IndexData")

# find all .json.bz2 files
files = list(folder.glob("*.json.bz2"))

print("Files found:")
for f in files:
    print(f.name)

# convert each file and save individual csvs
all_dfs = []

for file in files:
    print(f"Processing {file.name}...")
    
    df = json_bz2_to_long_table(file)
    all_dfs.append(df)
    
    output_file = folder / f"{file.stem.replace('.json', '')}_long.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved: {output_file.name}")

# combine all into one master dataframe
master_df = pd.concat(all_dfs, ignore_index=True)

master_output = folder / "all_indices_converted.csv"
master_df.to_csv(master_output, index=False)

print("\nDone.")
print(f"Combined dataset saved as: {master_output.name}")
print(master_df.head())
print(master_df.shape)
print(master_df["y"].describe())
print(master_df.isna().sum())

print(master_df.groupby("unique_id").size().describe())

print(master_df["unique_id"].unique())
print(master_df["unique_id"].nunique())
print(master_df["index_name"].value_counts())