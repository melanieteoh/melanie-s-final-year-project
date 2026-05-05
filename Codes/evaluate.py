import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_ROOT = Path("/Users/melanie/Desktop/final year project")
OUTPUT_DIR = PROJECT_ROOT / "Outputs"
EVAL_DIR = OUTPUT_DIR / "evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

INDICES = [
    "RUSSELL1000_return",
    "TOPIX1000_return",
]

MODEL_ORDER = [
    "TimeGPT",
    "FEDformer",
    "XGBoost",
    "ZeroBaseline",
    "LastReturnBaseline"
]

# maps each model's prediction column to a display name
MODEL_COLS = {
    "TimeGPT": "y_pred_timegpt",
    "FEDformer": "y_pred_fedformer",
    "XGBoost": "y_pred_xgboost",
    "ZeroBaseline": "y_pred_zero",
    "LastReturnBaseline": "y_pred_last",
}

# root mean squared error (RMSE)
def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

# measures whether predicted sign matches actual sign
# zero predictions excluded (won't have a bar in the output) because they don't have pos/neg direction
# direction matters more than magnitude 
def directional_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    mask = y_pred != 0.0
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return float("nan")
    
    return float((np.sign(y_true) == np.sign(y_pred)).mean())

# compute metrics (MAE, RMSE, directional accuracy) for one index pair and model
# returns None if no valid predictions can be evaluated
def compute_metrics(y_true: pd.Series, y_pred: pd.Series, model_name: str, unique_id: str) -> dict | None:
    # drop any rows where y_true is NaN but keep 0 predictions
    mask = y_true.notna() & y_pred.notna()
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        print(f"Warning: no valid predictions for {model_name} on {unique_id}.")
        return None

    return {
        "unique_id": unique_id,
        "model": model_name,
        "mae": float(mean_absolute_error(y_true, y_pred)), # lower better
        "rmse": rmse(y_true, y_pred), # lower better
        "directional_accuracy": directional_accuracy(y_true, y_pred), # higher better
        "n_predictions": int(len(y_true)),
    }

def load_predictions(model_dir: str, filename: str) -> pd.DataFrame | None:
    path = OUTPUT_DIR / model_dir / filename
    if not path.exists():
        print(f"NOT FOUND: {path}")
        return None

    df = pd.read_csv(path, parse_dates=["ds"])

    return df

# to make sure all models are evaluated only on dates where both TimeGPT and FEDformer produced valid predictions
def build_common_predictions() -> pd.DataFrame | None:
    timegpt_df = load_predictions("timegpt", "timegpt_predictions.csv")
    fedformer_df = load_predictions("fedformer", "fedformer_predictions.csv")
    xgboost_df = load_predictions("xgboost", "xgboost_predictions.csv")

    if timegpt_df is None or fedformer_df is None:
        print("Could not load all prediction files.")
        return None
    
    # fix possible naming issue in NeuralForecast
    if "y_pred_fedformer" not in fedformer_df.columns and "FEDformer" in fedformer_df.columns:
        fedformer_df = fedformer_df.rename(columns={"FEDformer": "y_pred_fedformer"})

    required_timegpt_cols = [
        "unique_id",
        "ds",
        "y_true",
        "y_pred_timegpt",
        "y_pred_zero",
        "y_pred_last",
    ]

    required_fedformer_cols = [
        "unique_id",
        "ds",
        "y_pred_fedformer",
    ]

    required_xgboost_cols = [
        "unique_id",
        "ds",
        "y_pred_xgboost",
    ]

    for col in required_timegpt_cols:
        if col not in timegpt_df.columns:
            raise ValueError(f"Missing column in TimeGPT predictions: {col}")

    for col in required_fedformer_cols:
        if col not in fedformer_df.columns:
            raise ValueError(f"Missing column in FEDformer predictions: {col}")

    for col in required_xgboost_cols:
        if col not in xgboost_df.columns:
            raise ValueError(f"Missing column in XGBoost predictions: {col}")
        
    timegpt_keep = timegpt_df[required_timegpt_cols].copy()
    fedformer_keep = fedformer_df[required_fedformer_cols].copy()
    xgboost_keep = xgboost_df[required_xgboost_cols].copy()

    # inner join keeps only dates where both TimeGPT and FEDformer have predictions
    common_df = timegpt_keep.merge(
        fedformer_keep,
        on=["unique_id", "ds"],
        how="inner",
    ).merge(
        xgboost_keep,
        on=["unique_id", "ds"],
        how="inner"
    )

    # drop rows where any model prediction is missing
    prediction_cols = [
        "y_true",
        "y_pred_timegpt",
        "y_pred_fedformer",
        "y_pred_xgboost",
        "y_pred_zero",
        "y_pred_last",
    ]

    common_df = common_df.dropna(subset=prediction_cols)

    common_df = common_df.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    print("\nCommon evaluation sample sizes:")
    print(common_df.groupby("unique_id").size())

    common_df.to_csv(EVAL_DIR / "common_predictions.csv", index=False)
    print(f"\nSaved: {EVAL_DIR / 'common_predictions.csv'}")

    return common_df

# plot grouped bar chart for each metric 
def plot_metrics_comparison(metrics_df: pd.DataFrame):
    metrics = ["mae", "rmse", "directional_accuracy"]
    titles  = {
        "mae": "Mean Absolute Error",
        "rmse": "Root Mean Squared Error",
        "directional_accuracy": "Directional Accuracy",
    }

    for metric in metrics:
        fig, axes = plt.subplots(1, len(INDICES), figsize=(5 * len(INDICES), 4), sharey=False)

        # handle case where only one index exists
        if len(INDICES) == 1:
            axes = [axes]

        for ax, unique_id in zip(axes, INDICES):
            subset = metrics_df[metrics_df["unique_id"] == unique_id].copy()

            if subset.empty:
                ax.set_title(f"{unique_id}\n(no data)")
                continue

            subset = subset.set_index("model").reindex(
                [m for m in MODEL_ORDER if m in subset["model"].values]
            ).reset_index()

            colors = {
                "TimeGPT": "#2196F3", # blue
                "FEDformer": "#4CAF50", # green
                "XGBoost": "#E91E63", # pink
                "ZeroBaseline": "#FF9800", # orange
                "LastReturnBaseline": "#9C27B0", # purple
            }

            # replace NaN with 0.0 only for plotting height
            plot_values = subset[metric].fillna(0.0)

            bars = ax.bar(
                subset["model"],
                plot_values,
                color=[colors.get(m, "#999999") for m in subset["model"]],
            )

            # add value on top of each bar
            for bar, actual_value in zip(bars, subset[metric]):
                if pd.isna(actual_value):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        0.0,
                        "N/A",
                        ha="center", va="bottom",
                        fontsize=8, color="#666666",
                    )
                else:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        f"{actual_value:.4f}",
                        ha="center", va="bottom",
                        fontsize=8,
                    )

            ax.set_title(unique_id.replace("_return", "").replace("_", " "))
            ax.set_ylabel(metric.upper().replace("_", " "))
            ax.tick_params(axis="x", rotation=30)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))

        fig.suptitle(titles[metric], fontsize=13, fontweight="bold")
        plt.tight_layout()

        save_path = EVAL_DIR / f"comparison_{metric}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path.name}")

# render all_metrics.csv into metrics_table
# ZeroBaseline da shows N/A
def render_metrics_table(metrics_df: pd.DataFrame):
    display = metrics_df.copy()

    zero_mask = display["model"] == "ZeroBaseline"
    display.loc[zero_mask, "directional_accuracy"] = float("nan")

    present_models = [m for m in MODEL_ORDER if m in display["model"].values]
    display = display.set_index(["unique_id", "model"]).reindex(
        pd.MultiIndex.from_product([INDICES, present_models], names=["unique_id", "model"])
    ).reset_index()

    display["unique_id"] = display["unique_id"].str.replace("_return", "", regex=False)
 
    display = display.rename(columns={
        "unique_id": "Index",
        "model": "Model",
        "mae": "MAE",
        "rmse": "RMSE",
        "directional_accuracy": "Directional Accuracy",
        "n_predictions": "n",
    })
 
    # round numbers, replace NaN DA with N/A
    def fmt(v, col):
        if pd.isna(v):
            return "N/A"
        if col == "n":
            return f"{int(v)}"
        return f"{v:.4f}"
 
    cell_text = []
    for i, row in display.iterrows():
        cell_text.append([
            row["Index"],
            row["Model"],
            fmt(row["MAE"], "MAE"),
            fmt(row["RMSE"], "RMSE"),
            fmt(row["Directional Accuracy"], "DA"),
            fmt(row["n"], "n"),
        ])
 
    columns = ["Index", "Model", "MAE", "RMSE", "Dir. Accuracy", "n"]
 
    fig, ax = plt.subplots(figsize=(12, 0.4 * (len(cell_text) + 1) + 0.8))
    ax.axis("off")
 
    table = ax.table(
        cellText=cell_text,
        colLabels=columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)
 
    n_cols = len(columns)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#37474F")
        else:
            row_idx = r - 1
            band = "#FFFFFF" if display.iloc[row_idx]["Index"] == INDICES[0].replace("_return", "") else "#F5F5F5"
            cell.set_facecolor(band)
 
    ax.set_title("Forecast Accuracy Metrics on Common Test Dates",
             fontsize=12, fontweight="bold", pad=12)
 
    save_path = EVAL_DIR / "metrics_table.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path.name}")

def main():
    common_df = build_common_predictions()

    if common_df is None or common_df.empty:
        print("No common prediction dates found.")
        return

    all_metrics = []

    for unique_id in common_df["unique_id"].unique():
        subset = common_df[common_df["unique_id"] == unique_id].copy()

        for model_name, pred_col in MODEL_COLS.items():
            if pred_col not in subset.columns:
                print(f"Missing column for {model_name}: {pred_col}")
                continue

            metric_row = compute_metrics(
                subset["y_true"],
                subset[pred_col],
                model_name,
                unique_id,
            )

            if metric_row:
                all_metrics.append(metric_row)

    if not all_metrics:
        print("No metrics produced.")
        return

    metrics_df = pd.DataFrame(all_metrics)

    metrics_df.to_csv(EVAL_DIR / "all_metrics.csv", index=False)

    print("\nEvaluation Results on Common Dates:")
    print(metrics_df.to_string(index=False))
    print(f"\nSaved: {EVAL_DIR / 'all_metrics.csv'}")

    print("\nGenerating plots...")
    plot_metrics_comparison(metrics_df)
    render_metrics_table(metrics_df)

    print("\nEvaluation complete.")

if __name__ == "__main__":
    main()