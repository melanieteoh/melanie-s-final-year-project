# Final Year Project

This repository contains the code for my final year project on financial time series forecasting using TimeGPT and FEDformer.

## Project Structure

```text
IndexData/
    splits/              # Train, validation and test CSV files
Outputs/
    timegpt/             # TimeGPT prediction outputs
    fedformer/           # FEDformer prediction outputs
run_TimeGPT.py           # Runs TimeGPT rolling prediction
run_FEDformer.py         # Runs FEDformer rolling prediction
evaluate.py              # Evaluates saved predictions
requirements.txt         # Python dependencies