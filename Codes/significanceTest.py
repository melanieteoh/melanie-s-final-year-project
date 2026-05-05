import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

PROJECT_ROOT = Path("/Users/melanie/Desktop/final year project")
EVAL_DIR = PROJECT_ROOT / "Outputs" / "evaluation"

INDICES = [
    "RUSSELL1000_return",
    "TOPIX1000_return",
]

MODEL_COLS = {
    "TimeGPT": "y_pred_timegpt",
    "FEDformer": "y_pred_fedformer",
    "XGBoost": "y_pred_xgboost",
    "ZeroBaseline": "y_pred_zero",
    "LastReturnBaseline": "y_pred_last",
}

# significance threshold for marking results as significant
ALPHA = 0.05

# binomial test on directional accuracy vs 0.5 (no-edge null)
# returns None when every prediction is exactly 0
def directional_significance(y_true: pd.Series, y_pred: pd.Series):
    # mask out exactly-zero predictions
    mask   = y_pred != 0.0
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    n = len(y_true)
    if n == 0:
        return None

    correct = int((np.sign(y_true) == np.sign(y_pred)).sum())

    # H0: P(correct sign) = 0.5
    # H1: > 0.5
    res = stats.binomtest(correct, n, p=0.5, alternative="greater")

    return {
        "n": n,
        "correct": correct,
        "directional_accuracy": correct / n,
        "p_value_one_sided": float(res.pvalue),
    }

# Wilcoxon signed-rank test on absolute errors
# pairs each prediction with the baseline's prediction for the same date
# H0: median(|err_model| - |err_baseline|) = 0
# H1: model has smaller errors than baseline
def magnitude_significance(y_true: pd.Series,
                           y_pred_model: pd.Series,
                           y_pred_baseline: pd.Series):
    err_model = (y_true - y_pred_model).abs()
    err_baseline = (y_true - y_pred_baseline).abs()

    res = stats.wilcoxon(err_model, err_baseline, alternative="less")

    return {
        "mae_model": float(err_model.mean()),
        "mae_baseline": float(err_baseline.mean()),
        "median_diff": float((err_model - err_baseline).median()),
        "p_value_one_sided": float(res.pvalue)
    }

# bar chart of directional accuracy per model and per index
# dashed line at 0.5 marks coin-flip baseline
# ZeroBaseline is N/A because no da
# asterisk above bar is significantly > 0.5 at p < alpha

def plot_directional_significance(direction_df: pd.DataFrame):
    colors = {
        "TimeGPT": "#2196F3",
        "FEDformer": "#4CAF50",
        "XGBoost": "#E91E63",
        "ZeroBaseline": "#FF9800",
        "LastReturnBaseline": "#9C27B0",
    }
 
    fig, axes = plt.subplots(1, len(INDICES), figsize=(5 * len(INDICES), 4.2),
                             sharey=True)
    if len(INDICES) == 1:
        axes = [axes]
 
    for ax, unique_id in zip(axes, INDICES):
        sub = direction_df[direction_df["unique_id"] == unique_id]
        models = sub["model"].tolist()
        das = sub["directional_accuracy"].fillna(0.0).values
        ps = sub["p_value_one_sided"].values
 
        bar_colors = [colors.get(m, "#999999") for m in models]
        bars = ax.bar(models, das, color=bar_colors)
 
        # chance line
        ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.6,
                   label="Chance (0.5)")
 
        # label each bar
        for bar, da, p, m in zip(bars, das, ps, models):
            if pd.isna(p) or m == "ZeroBaseline":
                ax.text(bar.get_x() + bar.get_width() / 2, 0.02,
                        "N/A", ha="center", va="bottom",
                        fontsize=8, color="#666666")
            else:
                # value
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{da:.4f}", ha="center", va="bottom", fontsize=8)
                # significance marker above value
                if p < ALPHA:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.035,
                            "*", ha="center", va="bottom",
                            fontsize=14, fontweight="bold", color="black")
 
        ax.set_title(unique_id.replace("_return", "").replace("_", " "))
        ax.set_ylabel("Directional Accuracy")
        ax.set_ylim(0, 0.65)
        ax.tick_params(axis="x", rotation=30)
        ax.legend(loc="lower right", fontsize=8)
 
    fig.suptitle(f"Directional Accuracy vs Chance  (* = p < {ALPHA})",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
 
    save_path = EVAL_DIR / "significance_directional.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path.name}")
 
# grid showing whether each main model significantly beats each baseline on MAE
# green: model error significantly less than baseline error (p < alpha)
# red: model error greater than baseline error (1-p < alpha)
# grey: no significant difference
# cell text shows the one sided p value
def plot_magnitude_significance(magnitude_df: pd.DataFrame):
    main_models = ["TimeGPT", "FEDformer", "XGBoost"]
    baselines = ["ZeroBaseline", "LastReturnBaseline"]
 
    fig, axes = plt.subplots(1, len(INDICES), figsize=(5 * len(INDICES), 4.5),
                             sharey=True)
    if len(INDICES) == 1:
        axes = [axes]
 
    for ax, unique_id in zip(axes, INDICES):
        sub = magnitude_df[magnitude_df["unique_id"] == unique_id]
 
        # build cell colours and text
        cell_colours = []
        cell_text = []
        for model in main_models:
            row_colours = []
            row_text = []
            for baseline in baselines:
                rec = sub[(sub["model"] == model) & (sub["baseline"] == baseline)]
                if rec.empty:
                    row_colours.append("#EEEEEE")
                    row_text.append("—")
                    continue
                p = float(rec["p_value_one_sided"].iloc[0])
                # one-sided test: H1 is "model < baseline"
                # small p = model significantly better
                # p close to 1 = model significantly worse (1 - p < ALPHA)
                if p < ALPHA:
                    row_colours.append("#A5D6A7")  # green: better
                    row_text.append(f"better\np = {p:.2e}")
                elif (1 - p) < ALPHA:
                    row_colours.append("#EF9A9A")  # red: worse
                    row_text.append(f"worse\np = {p:.2e}")
                else:
                    row_colours.append("#E0E0E0")  # grey: no difference
                    row_text.append(f"tie\np = {p:.3f}")
            cell_colours.append(row_colours)
            cell_text.append(row_text)
 
        # draw the grid
        ax.set_xlim(0, len(baselines))
        ax.set_ylim(0, len(main_models))
        ax.invert_yaxis()  # so first model is on top
 
        for i, model in enumerate(main_models):
            for j, baseline in enumerate(baselines):
                ax.add_patch(plt.Rectangle((j, i), 1, 1,
                                           facecolor=cell_colours[i][j],
                                           edgecolor="white", linewidth=2))
                ax.text(j + 0.5, i + 0.5, cell_text[i][j],
                        ha="center", va="center", fontsize=9)
 
        # axis labels
        ax.set_xticks([j + 0.5 for j in range(len(baselines))])
        ax.set_xticklabels(baselines, rotation=20)
        ax.set_yticks([i + 0.5 for i in range(len(main_models))])
        ax.set_yticklabels(main_models)
        ax.tick_params(axis="both", which="both", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
 
        ax.set_title(unique_id.replace("_return", "").replace("_", " "))
 
    fig.suptitle(f"Magnitude Significance: Model vs Baseline  "
                 f"(green = model better, red = worse, p < {ALPHA})",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
 
    save_path = EVAL_DIR / "significance_magnitude.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path.name}")

def main():
    common_df = pd.read_csv(EVAL_DIR / "common_predictions.csv", parse_dates=["ds"])

    direction_rows = []
    magnitude_rows = []

    for unique_id in common_df["unique_id"].unique():
        sub = common_df[common_df["unique_id"] == unique_id]

        # directional accuracy significance for every model
        for model_name, col in MODEL_COLS.items():
            res = directional_significance(sub["y_true"], sub[col])
            if res is None:
                # for ZeroBaseline because its directional accuracy is undefined
                direction_rows.append({
                    "unique_id": unique_id,
                    "model": model_name,
                    "n": 0,
                    "correct": 0,
                    "directional_accuracy": float("nan"),
                    "p_value_one_sided": float("nan"),
                })
            else:
                direction_rows.append({"unique_id": unique_id,
                                       "model": model_name,
                                       **res})

        # magnitude significance
        # compare each main model against each baseline
        for model_name in ["TimeGPT", "FEDformer", "XGBoost"]:
            for baseline_name in ["ZeroBaseline", "LastReturnBaseline"]:
                res = magnitude_significance(
                    sub["y_true"],
                    sub[MODEL_COLS[model_name]],
                    sub[MODEL_COLS[baseline_name]],
                )
                magnitude_rows.append({
                    "unique_id": unique_id,
                    "model": model_name,
                    "baseline": baseline_name,
                    **res,
                })

    direction_df = pd.DataFrame(direction_rows)
    magnitude_df = pd.DataFrame(magnitude_rows)

    direction_df.to_csv(EVAL_DIR / "significance_directional.csv", index=False)
    magnitude_df.to_csv(EVAL_DIR / "significance_magnitude.csv", index=False)

    print("Directional accuracy: one-sided binomial vs 0.5")
    print(direction_df.to_string(index=False))
    print("\nMagnitude: Wilcoxon |model err| < |baseline err| (one-sided)")
    print(magnitude_df.to_string(index=False))
    print("\nSaved to significance_directional.csv, significance_magnitude.csv")

    print("\nGenerating plots...")
    plot_directional_significance(direction_df)
    plot_magnitude_significance(magnitude_df)
 
    print("\nSignificance testing complete.")

if __name__ == "__main__":
    main()