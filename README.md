# Final Year Project

## Forecasting Returns using Time-Series Models and SHAP
This repository contains the code for my final year project on short-horizon financial return forecasting and explainability. The project compares TimeGPT, FEDformer and XGBoost against two simple baseline models, ZeroBaseline and LastReturnBaseline. It also applies SHAP to the trained XGBoost models to explain which return-derived features influence predictions.

The project focuses on daily return forecasting for two indices:
- RUSSELL1000
- TOPIX1000

## Project Aim

The aim of this project is to build a reproducible forecasting pipeline that combines:

- financial return data preprocessing
- short-horizon forecasting
- baseline model comparison
- performance evaluation
- statistical significance testing
- directional stability analysis
- SHAP-based explainability

## Project Structure

```text
IndexData/
    RUSSELL1000_long.csv              # converted long-format RUSSELL1000 data
    TOPIX1000_long.csv                # converted long-format TOPIX1000 data

    RUSSELL1000_returns_cleaned.csv   # cleaned RUSSELL1000 return series
    TOPIX1000_returns_cleaned.csv     # cleaned TOPIX1000 return series

    all_indices_converted.csv         # combined converted index dataset
    all_indices_cleaned.csv           # combined cleaned return dataset used for feature engineering

    splits/                           # train, validation and test CSV files

Outputs/
    evaluation/              # evaluation tables and plots
    fedformer/               # FEDformer prediction outputs
    shap/                    # SHAP plots, feature importance and ablation results
    timegpt/                 # TimeGPT prediction outputs
    xgboost/                 # XGBoost prediction outputs and saved models

Codes/
    convertIndexData.py      # converts raw index data into usable format
    dataClean.py             # cleans and aligns index return data
    prepareData.py           # creates lag, volatility and momentum features, then splits data
    run_TimeGPT.py           # runs TimeGPT rolling one-step-ahead forecasts
    run_FEDformer.py         # trains and evaluates FEDformer
    run_XGBoost.py           # trains XGBoost and saves predictions/models
    evaluate.py              # evaluates all model predictions on common dates
    significanceTest.py      # runs binomial and Wilcoxon significance tests
    stabilityCheck.py        # checks directional accuracy stability across sub-periods
    run_SHAP.py              # generates SHAP global/local explanations and ablation results

requirements.txt             # Python dependencies
README.md                    # project instructions