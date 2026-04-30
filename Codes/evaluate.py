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

# maps each model's prediction column to a display name
# add new models here if you extend the pipeline later
MODEL_COLS = {
    "TimeGPT": "y_pred_timegpt",
    "FEDformer": "y_pred_fedformer",
    "fedformer": "y_pred_fedformer",
    "ZeroBaseline": "y_pred_zero",
    "LastReturnBaseline": "y_pred_last",
}

INDICES = [
    "RUSSELL1000_return",
    "TOPIX1000_return",
]

# root mean squared error (RMSE)
def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

# fraction of predictions where the sign matches actual sign
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
def compute_metrics(y_true: pd.Series, y_pred: pd.Series,
                    model_name: str, unique_id: str) -> dict | None:
    # drop any rows where y_true is NaN but keep 0 predictions
    mask = y_true.notna() & y_pred.notna()
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        print(f"Warning: no valid predictions for {model_name} on {unique_id}.")
        return None

    da = directional_accuracy(y_true, y_pred)

    return {
        "unique_id": unique_id,
        "model": model_name,
        "mae": float(mean_absolute_error(y_true, y_pred)), # lower better
        "rmse": rmse(y_true, y_pred), # lower better
        "directional_accuracy": da, # higher better
        "n_predictions": int(len(y_true)),
    }

def load_predictions(model_dir: str, filename: str) -> pd.DataFrame | None:
    path = OUTPUT_DIR / model_dir / filename
    if not path.exists():
        print(f"NOT FOUND: {path}")
        return None

    df = pd.read_csv(path, parse_dates=["ds"])
    return df

# plot grouped bar chart for each metric 
def plot_metrics_comparison(metrics_df: pd.DataFrame):
    metrics = ["mae", "rmse", "directional_accuracy"]
    titles  = {
        "mae": "Mean Absolute Error",
        "rmse": "Root Mean Squared Error",
        "directional_accuracy": "Directional Accuracy",
    }

    for metric in metrics:
        fig, axes = plt.subplots(1, len(INDICES), figsize=(5 * len(INDICES), 4),
                                 sharey=False)

        # handle case where only one index exists
        if len(INDICES) == 1:
            axes = [axes]

        for ax, unique_id in zip(axes, INDICES):
            subset = metrics_df[metrics_df["unique_id"] == unique_id]

            if subset.empty:
                ax.set_title(f"{unique_id}\n(no data)")
                continue

            # sort so baselines appear after the main models
            model_order = ["TimeGPT", "FEDformer", "ZeroBaseline", "LastReturnBaseline"]
            subset = subset.set_index("model").reindex(
                [m for m in model_order if m in subset["model"].values]
            ).reset_index()

            colors = {
                "TimeGPT": "#2196F3", # blue
                "FEDformer": "#4CAF50", # green
                "ZeroBaseline": "#FF9800", # orange
                "LastReturnBaseline": "#FFFF00", # yellow
            }

            bars = ax.bar(
                subset["model"],
                subset[metric],
                color=[colors.get(m, "#999999") for m in subset["model"]]
            )

            # add value labels on top of each bar
            for bar, model_name in zip(bars, subset["model"]):
                height = bar.get_height()

                # NaN bars have height of 0
                if np.isnan(height) or (model_name == "ZeroBaseline" and height == 0.0):
                    # label N/A instead of 0.0000
                    label_text = "N/A"
                else:
                    label_text = f"{height:.4f}"

                ax.text(bar.get_x() + bar.get_width() / 2, height,
                        f"{height:.4f}", ha="center", va="bottom", fontsize=8)

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

# time-series plot of predicted vs actual returns for each index to show all models on same axes
def plot_predictions_vs_actual(combined_df: pd.DataFrame):
    for unique_id in combined_df["unique_id"].unique():
        subset = combined_df[combined_df["unique_id"] == unique_id].sort_values("ds")

        fig, ax = plt.subplots(figsize=(14, 4))

        # actual returns as a thin black line for reference
        ax.plot(subset["ds"], subset["y_true"],
                color="black", linewidth=0.8, label="Actual", zorder=5)

        # plot whichever prediction columns are present in this file
        plot_map = {
            "y_pred_timegpt": ("TimeGPT", "#2196F3"),
            "y_pred_fedformer":("FEDformer", "#4CAF50"),
            "y_pred_zero": ("ZeroBaseline", "#FF9800"),
            "y_pred_last": ("LastReturnBaseline", "#FFFF00"),
        }
        for col, (label, colour) in plot_map.items():
            if col in subset.columns:
                ax.plot(subset["ds"], subset[col],
                        linewidth=0.6, alpha=0.7, label=label, color=colour)

        ax.axhline(0, color="black", linewidth=0.4, linestyle="--")
        ax.set_title(f"{unique_id.replace('_return','').replace('_',' ')} — Predicted vs Actual Returns")
        ax.set_xlabel("Date")
        ax.set_ylabel("Daily Return")
        ax.legend(fontsize=8)

        plt.tight_layout()
        save_path = EVAL_DIR / f"predictions_{unique_id}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path.name}")

def main():
    all_metrics = []
    all_predictions = []

    # TimeGPT predictions
    timegpt_df = load_predictions("timegpt", "timegpt_predictions.csv")
    if timegpt_df is not None:
        all_predictions.append(timegpt_df)

        for unique_id in timegpt_df["unique_id"].unique():
            subset = timegpt_df[timegpt_df["unique_id"] == unique_id]

            for model_name, pred_col in [
                ("TimeGPT", "y_pred_timegpt"),
                ("ZeroBaseline", "y_pred_zero"),
                ("LastReturnBaseline", "y_pred_last"),
            ]:
                if pred_col not in subset.columns:
                    continue
                m = compute_metrics(subset["y_true"], subset[pred_col],
                                    model_name, unique_id)
                if m:
                    all_metrics.append(m)

    # load FEDformer predictions
    fedformer_df = load_predictions("fedformer", "fedformer_predictions.csv")
    if fedformer_df is not None:
        print("FEDformer columns:", fedformer_df.columns.tolist())
        print("FEDformer rows:", len(fedformer_df))

        if "y_pred_fedformer" not in fedformer_df.columns and "FEDformer" in fedformer_df.columns:
            fedformer_df = fedformer_df.rename(columns={"FEDformer": "y_pred_fedformer"})
        
        all_predictions.append(fedformer_df)

        for unique_id in fedformer_df["unique_id"].unique():
            subset = fedformer_df[fedformer_df["unique_id"] == unique_id]

            for model_name, pred_col in [
                ("fedformer", "y_pred_fedformer"),
                ("ZeroBaseline", "y_pred_zero"),
                ("LastReturnBaseline", "y_pred_last"),
            ]:
                if pred_col not in subset.columns:
                    print(f"Missing column for {model_name}: {pred_col}")
                    continue
                m = compute_metrics(subset["y_true"], subset[pred_col], model_name, unique_id)
                if m:
                    all_metrics.append(m)

    if not all_metrics:
        print("No metrics produced.")
        return

    # save combined metrics table
    metrics_df = pd.DataFrame(all_metrics)

    # deduplicate baselines: ZeroBaseline and LastReturn appear in both model
    # output files with identical values and keep only the first occurrence
    metrics_df = metrics_df.drop_duplicates(subset=["unique_id", "model"])

    metrics_df.to_csv(EVAL_DIR / "all_metrics.csv", index=False)
    print("\nEvaluation Results:")
    print(metrics_df.to_string(index=False))
    print(f"\nSaved: {EVAL_DIR / 'all_metrics.csv'}")

    print("\nGenerating plots...")
    plot_metrics_comparison(metrics_df)

    if all_predictions:
        # merge TimeGPT and FEDformer predictions on the shared columns
        combined = all_predictions[0]
        for df in all_predictions[1:]:
            new_cols = [c for c in df.columns
                        if c not in combined.columns or c in ["unique_id", "ds"]]
            combined = combined.merge(
                df[["unique_id", "ds"] + [c for c in new_cols
                                          if c not in ["unique_id", "ds"]]],
                on=["unique_id", "ds"],
                how="outer",
            )
        plot_predictions_vs_actual(combined)

    print("\nEvaluation complete.")

if __name__ == "__main__":
    main()