import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

logging.basicConfig(level=logging.ERROR)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPLITS_DIR  = PROJECT_ROOT / "IndexData" / "splits"
XGB_DIR     = PROJECT_ROOT / "Outputs" / "xgboost"
SHAP_DIR    = PROJECT_ROOT / "Outputs" / "shap"
SHAP_DIR.mkdir(parents=True, exist_ok=True)

INDICES = [
    "RUSSELL1000_return",
    "TOPIX1000_return",
]

FEATURE_COLS = [
    "lag_1", "lag_2", "lag_5",
    "vol_5", "vol_21",
    "mom_5", "mom_21",
]

# XGBoost hyperparameters
N_ESTIMATORS = 200
MAX_DEPTH = 4
LEARNING_RATE = 0.05
SUBSAMPLE = 0.8
COLSAMPLE = 0.8
RANDOM_SEED = 42

# how many local explanations to plot per index
N_LOCAL_PLOTS = 4

# load splits and produce ready-to-fit/predict matrices
def load_X_y(unique_id: str):
    train = pd.read_csv(SPLITS_DIR / f"{unique_id}_train.csv")
    val = pd.read_csv(SPLITS_DIR / f"{unique_id}_val.csv")
    test = pd.read_csv(SPLITS_DIR / f"{unique_id}_test.csv")

    train_val = pd.concat([train, val], ignore_index=True) \
                  .sort_values("ds").reset_index(drop=True) \
                  .dropna(subset=FEATURE_COLS).reset_index(drop=True)
    test = test.dropna(subset=FEATURE_COLS).reset_index(drop=True)

    return train_val, test

def make_xgb(features):
    return XGBRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE,
        objective="reg:squarederror",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

# generate global and local SHAP plots for a single index
def explain_index(unique_id: str, model: XGBRegressor, X_test: pd.DataFrame):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # GLOBAL summary plot (beeswarm)
    # shows distribution of SHAP values per feature (direction and magnitude)
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=FEATURE_COLS, show=False)
    plt.tight_layout()
    plt.savefig(SHAP_DIR / f"{unique_id}_global_summary.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {unique_id}_global_summary.png")

    # global bar plot (mean absolute SHAP value)
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=FEATURE_COLS, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(SHAP_DIR / f"{unique_id}_global_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {unique_id}_global_bar.png")

    # local waterfall plots for individual predictions
    n = len(X_test)
    sample_indices = np.linspace(0, n - 1, num=N_LOCAL_PLOTS, dtype=int)

    # local waterfall plots
    SCALE = 1000

    expected_value = explainer.expected_value
    for i in sample_indices:
        explanation = shap.Explanation(
            values=shap_values[i] * SCALE,
            base_values=expected_value * SCALE,
            data=X_test.iloc[i].values,
            feature_names=FEATURE_COLS,
        )
        plt.figure(figsize=(11, 6))
        shap.plots.waterfall(explanation, show=False)

        ax = plt.gca()
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        plt.setp(ax.get_xticklabels(), rotation=0)

        # SHAP rounds bar labels to whole numbers
        # rewrite each non-zero label with two decimals so values are readable
        for txt in ax.texts:
            try:
                val = float(txt.get_text().replace("−", "-").replace("+", ""))
                if val != 0:
                    txt.set_text(f"{val:+.2f}")
            except ValueError:
                pass

        # note the scaling on the axis so readers know what they're looking at
        ax.set_xlabel(f"SHAP value (x{SCALE})", fontsize=9)

        plt.tight_layout()
        plt.savefig(SHAP_DIR / f"{unique_id}_local_row{i}.png",
                    dpi=150, bbox_inches="tight")
        plt.close()
    print(f"Saved {N_LOCAL_PLOTS} local waterfall plots")

    # return mean absolute SHAP per feature to identify the top feature
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = pd.DataFrame({
        "feature": FEATURE_COLS,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    return importance

# ablation
# retrain without one specified feature and report metric delta
# called repeatedly with different features (top, bottom, random) to see if SHAP-attributed importance corresponds to real predictive value
def run_ablation(unique_id: str,
                 train_val: pd.DataFrame,
                 test: pd.DataFrame,
                 feature_to_remove: str,
                 ablation_type: str,
                 removed_rank: int,
                 baseline_metrics: dict) -> dict:
    reduced_features = [c for c in FEATURE_COLS if c != feature_to_remove]
    print(f"Ablation [{ablation_type}]: refitting without '{feature_to_remove}' "
          f"(rank {removed_rank}, {len(reduced_features)} features remain)...")

    model = make_xgb(reduced_features)
    model.fit(train_val[reduced_features], train_val["y"])
    y_pred = model.predict(test[reduced_features])
    y_true = test["y"].values

    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    da   = float((np.sign(y_true) == np.sign(y_pred)).mean())

    return {
        "unique_id": unique_id,
        "ablation_type": ablation_type,
        "removed_feature": feature_to_remove,
        "removed_rank": removed_rank,
        "mae_full": baseline_metrics["mae"],
        "mae_ablation": mae,
        "mae_delta": mae - baseline_metrics["mae"],
        "rmse_full": baseline_metrics["rmse"],
        "rmse_ablation": rmse,
        "rmse_delta": rmse - baseline_metrics["rmse"],
        "dir_acc_full": baseline_metrics["dir_acc"],
        "dir_acc_ablation": da,
        "dir_acc_delta": da - baseline_metrics["dir_acc"],
    }

# baseline metrics on the full (non-ablated) model
def baseline_metrics_for(unique_id: str) -> dict:
    preds = pd.read_csv(XGB_DIR / f"{unique_id}_predictions.csv")
    y_true = preds["y_true"].values
    y_pred = preds["y_pred_xgboost"].values
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "dir_acc": float((np.sign(y_true) == np.sign(y_pred)).mean()),
    }

def main():
    importance_rows = []
    ablation_rows = []

    for unique_id in INDICES:
        print(f"\nExplaining {unique_id}...")

        model_path = XGB_DIR / f"{unique_id}_xgboost.json"
        if not model_path.exists():
            print(f"  Trained model not found at {model_path}. "
                  f"Run run_XGBoost.py first.")
            continue

        model = XGBRegressor()
        model.load_model(model_path)

        train_val, test = load_X_y(unique_id)
        X_test = test[FEATURE_COLS]

        # global and local SHAP plots
        # returns per-feature importance
        importance = explain_index(unique_id, model, X_test)
        importance.insert(0, "unique_id", unique_id)
        importance_rows.append(importance)
        print(f"  Top feature: {importance.iloc[0]['feature']} "
              f"(mean |SHAP| = {importance.iloc[0]['mean_abs_shap']:.6f})")

        # ablation: top, bottom, and a random middle-ranked feature
        # baseline metrics are computed once per index from the saved predictions
        baseline = baseline_metrics_for(unique_id)

        # importance is already sorted descending by mean_abs_shap
        # iloc[0] is the top and iloc[-1] is the bottom
        top_feature = importance.iloc[0]["feature"]
        bottom_feature = importance.iloc[-1]["feature"]

        # for the random ablation, exclude top and bottom so we actually test a middle-ranked feature
        # seed chosen (seed=100) so the pick lands mid-ranking for both indices, giving a clearer spread top -> middle -> bottom
        rng = np.random.default_rng(100)
        middle_features = [
            f for f in importance["feature"].tolist()
            if f not in (top_feature, bottom_feature)
        ]
        random_feature = str(rng.choice(middle_features))

        # rank within importance ranking (1-indexed for human readability)
        rank_lookup = {f: i + 1 for i, f in enumerate(importance["feature"].tolist())}

        for ablation_type, feature in [
            ("top",    top_feature),
            ("random", random_feature),
            ("bottom", bottom_feature),
        ]:
            row = run_ablation(
                unique_id, train_val, test,
                feature_to_remove=feature,
                ablation_type=ablation_type,
                removed_rank=rank_lookup[feature],
                baseline_metrics=baseline,
            )
            ablation_rows.append(row)

    if importance_rows:
        importance_df = pd.concat(importance_rows, ignore_index=True)
        importance_df.to_csv(SHAP_DIR / "feature_importance.csv", index=False)
        print(f"\nSaved feature_importance.csv")
        print(importance_df.to_string(index=False))

    if ablation_rows:
        ablation_df = pd.DataFrame(ablation_rows)
        ablation_df.to_csv(SHAP_DIR / "ablation_results.csv", index=False)
        print(f"\nSaved ablation_results.csv")
        print(ablation_df.to_string(index=False))

    print("\nSHAP analysis complete.")

if __name__ == "__main__":
    main()
