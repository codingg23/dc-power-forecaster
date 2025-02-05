"""
evaluate.py

Evaluation script for trained forecasting models.
Computes MAE, RMSE, MAPE and plots residuals.
"""
import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from data.preprocessing import PowerDataset, PreprocessConfig
from models.lstm_forecaster import LSTMForecaster

def mae(y_true, y_pred): return np.mean(np.abs(y_true - y_pred))
def rmse(y_true, y_pred): return np.sqrt(np.mean((y_true - y_pred) ** 2))
def mape(y_true, y_pred):
    mask = y_true > 1.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="./eval_results/")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = LSTMForecaster(ckpt["config"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    pp = PowerDataset(PreprocessConfig(horizon=ckpt["config"].horizon))
    df = pp.load_and_preprocess(args.data)
    _, _, test_df = pp.train_val_test_split(df)
    pp.fit_scaler(df.iloc[:int(len(df)*0.7)])  # refit on train portion

    arr = pp.transform(test_df)
    X, y = pp.make_windows(arr)

    with torch.no_grad():
        preds = model(torch.from_numpy(X)).numpy()

    # inverse transform
    y_true = np.array([pp.inverse_transform_target(y[i]) for i in range(len(y))])
    y_pred = np.array([pp.inverse_transform_target(preds[i]) for i in range(len(preds))])

    print(f"MAE:  {mae(y_true.flatten(), y_pred.flatten()):.2f} kW")
    print(f"RMSE: {rmse(y_true.flatten(), y_pred.flatten()):.2f} kW")
    print(f"MAPE: {mape(y_true.flatten(), y_pred.flatten()):.2f}%")

    Path(args.output).mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"y_true": y_true.flatten(), "y_pred": y_pred.flatten()}).to_csv(
        f"{args.output}/predictions.csv", index=False
    )

if __name__ == "__main__":
    main()
