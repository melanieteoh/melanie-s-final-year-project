import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path("/Users/melanie/Desktop/final year project")
EVAL_DIR = PROJECT_ROOT / "Outputs" / "evaluation"

INDICES = [
    "RUSSELL1000_return",
    "TOPIX1000_return"
]

# ZeroBaseline excluded because directional accuracy is undefined when prediction is 0
MODEL_COLS = {
    "TimeGPT": "y_pred_timegpt",
    "FEDformer": "y_pred_fedformer",
    "XGBoost": "y_pred_xgboost",
    "LastReturnBaseline": "y_pred_last",
}

# 4 chronological chunks to split the test period into which gives around 125-130 obs per chunk for the splits
N_SPLITS = 4

# directional accuracy on a single chunk
def directional_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    mask   = y_pred != 0.0
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return float("nan")
    return float((np.sign(y_true) == np.sign(y_pred)).mean())

def main():
    common_df = pd.read_csv(EVAL_DIR / "common_predictions.csv", parse_dates=["ds"])

    rows = []
    for unique_id in INDICES:
        sub = common_df[common_df["unique_id"] == unique_id] \
                       .sort_values("ds").reset_index(drop=True)

        # split into N equal-sized chronological chunks by row index 
        # use row index to keep each chunk the same size
        # doesn't use calendar days
        sub["split"] = pd.qcut(
            sub.index, N_SPLITS,
            labels=[f"Q{i+1}" for i in range(N_SPLITS)],
        )

        for split_label, chunk in sub.groupby("split", observed=False):
            for model_name, col in MODEL_COLS.items():
                rows.append({
                    "unique_id": unique_id,
                    "split": str(split_label),
                    "model": model_name,
                    "directional_accuracy": directional_accuracy(
                        chunk["y_true"], chunk[col]
                    ),
                    "n": len(chunk),
                    "start_date": chunk["ds"].min().date(),
                    "end_date":   chunk["ds"].max().date(),
                })

    stab_df = pd.DataFrame(rows)
    stab_df.to_csv(EVAL_DIR / "stability_directional.csv", index=False)
    print(stab_df.to_string(index=False))

    # one subplot per index, lines per model
    # x = split, y = directional accuracy
    fig, axes = plt.subplots(1, len(INDICES), figsize=(5 * len(INDICES), 4))
    if len(INDICES) == 1:
        axes = [axes]

    colors = {
        "TimeGPT": "#2196F3",
        "FEDformer": "#4CAF50",
        "XGBoost": "#E91E63",
        "LastReturnBaseline": "#9C27B0"
    }

    for ax, unique_id in zip(axes, INDICES):
        sub = stab_df[stab_df["unique_id"] == unique_id]
        for model_name in MODEL_COLS:
            mdata = sub[sub["model"] == model_name]
            ax.plot(mdata["split"], mdata["directional_accuracy"],
                    marker="o", label=model_name,
                    color=colors[model_name])

        # 0.5 to reflect coin flip
        ax.axhline(0.5, color="black", linewidth=0.8, linestyle="--",
                   label="Coin flip (0.5)")

        # build x labels with date ranges so each split shows its calendar window
        x_labels = []
        for split in [f"Q{i+1}" for i in range(N_SPLITS)]:
            chunk = sub[sub["split"] == split].iloc[0]
            x_labels.append(f"{split}\n{chunk['start_date']}\nto {chunk['end_date']}")
        ax.set_xticks(range(N_SPLITS))
        ax.set_xticklabels(x_labels, fontsize=7)

        ax.set_title(unique_id.replace("_return", "").replace("_", " "))
        ax.set_ylabel("Directional Accuracy")
        ax.set_xlabel(f"Test period ({N_SPLITS} chronological splits)")
        ax.set_ylim(0.35, 0.65)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    stability_summary = stab_df.groupby(["unique_id", "model"])["directional_accuracy"] \
                             .agg(["mean", "std", "min", "max"]).reset_index()
    stability_summary.to_csv(EVAL_DIR / "stability_summary.csv", index=False)
    print("\nStability summary:")
    print(stability_summary.to_string(index=False))

    # build summary stats per model per index
    summary = stab_df.groupby(["unique_id", "model"])["directional_accuracy"] \
                     .agg(["mean", "std", "min", "max"]).reset_index()
    summary.to_csv(EVAL_DIR / "stability_summary.csv", index=False)
    print("\nStability summary:")
    print(summary.to_string(index=False))

    # render as a table image
    summary_display = summary.copy()
    summary_display["unique_id"] = summary_display["unique_id"].str.replace("_return", "", regex=False)
    summary_display = summary_display.rename(columns={
        "unique_id": "Index", "model": "Model",
        "mean": "Mean DA", "std": "Std DA",
        "min": "Min DA", "max": "Max DA",
    })

    cell_text = [[row["Index"], row["Model"],
                  f"{row['Mean DA']:.4f}", f"{row['Std DA']:.4f}",
                  f"{row['Min DA']:.4f}", f"{row['Max DA']:.4f}"]
                 for _, row in summary_display.iterrows()]

    fig, ax = plt.subplots(figsize=(10, 0.4 * (len(cell_text) + 1) + 0.8))
    ax.axis("off")
    ax.set_position([0, 0, 1, 0.85])
    table = ax.table(cellText=cell_text,
                     colLabels=summary_display.columns.tolist(),
                     loc="center", cellLoc="center", colLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#37474F")
        else:
            band = "#FFFFFF" if cell_text[r-1][0] == "RUSSELL1000" else "#F5F5F5"
            cell.set_facecolor(band)

    ax.set_title("Stability of Directional Accuracy Across Test Period",
                 fontsize=12, fontweight="bold", pad=12)
    plt.savefig(EVAL_DIR / "stability_summary.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved: stability_summary.png")

    fig.suptitle("Directional Accuracy Stability Across Test Period",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(EVAL_DIR / "stability_directional.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("\nSaved to stability_directional.csv, stability_directional.png")

if __name__ == "__main__":
    main()