# generate ablation table from ablation_results.csv

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path("/Users/melanie/Desktop/final year project")
SHAP_DIR = PROJECT_ROOT / "Outputs" / "shap"

INDICES = [
    "RUSSELL1000_return",
    "TOPIX1000_return",
]

ABLATION_ORDER = ["top", "random", "bottom"]

def main():
    df = pd.read_csv(SHAP_DIR/ "ablation_results.csv")

    # display columns: index, ablation type, removed feature (rank), MAE delta, RMSE delta, DA delta
    rows = []
    for unique_id in INDICES:
        for ablation_type in ABLATION_ORDER:
            rec = df[(df["unique_id"] == unique_id) &
                     (df["ablation_type"] == ablation_type)]
            if rec.empty:
                continue
            r = rec.iloc[0]
            rows.append([
                unique_id.replace("_return", ""),
                ablation_type.title(),
                f"{r['removed_feature']} (rank {int(r['removed_rank'])})",
                f"{r['mae_delta']:+.5f}",
                f"{r['rmse_delta']:+.5f}",
                f"{r['dir_acc_delta']:+.4f}",
            ])

    columns = ["Index", "Ablation", "Removed Feature",
               "Δ MAE", "Δ RMSE", "Δ DA"]

    fig, ax = plt.subplots(figsize=(11, 0.4 * (len(rows) + 1) + 0.8))
    ax.axis("off")
    ax.set_position([0, 0, 1, 0.85])

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#37474F")
        else:
            row_idx = r - 1
            band = "#FFFFFF" if rows[row_idx][0] == "RUSSELL1000" else "#F5F5F5"
            cell.set_facecolor(band)

    ax.set_title("Ablation Study: XGBoost Performance with Feature Removed",
                 fontsize=12, fontweight="bold", pad=12)

    save_path = SHAP_DIR / "ablation_table.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")
    print("\nAblation table:")
    print(pd.DataFrame(rows, columns=columns).to_string(index=False))

if __name__ == "__main__":
    main()