"""
predict.py  -  run inference with a trained checkpoint

Usage:
    python predict.py --checkpoint ./checkpoints/lstm_best.pt \
                      --data ./data/recent_power.parquet \
                      --output ./predictions/forecast.json
"""
import argparse, json, torch
import numpy as np
import pandas as pd
from pathlib import Path
from data.preprocessing import PowerDataset, PreprocessConfig
from models.lstm_forecaster import LSTMForecaster

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--scaler", default=None)
    parser.add_argument("--output", default="./predictions/forecast.json")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = LSTMForecaster(ckpt["config"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    pp = PowerDataset(PreprocessConfig(horizon=ckpt["config"].horizon, context_len=ckpt["config"].context_len))
    df = pp.load_and_preprocess(args.data)

    scaler_path = args.scaler or str(Path(args.checkpoint).parent / "scaler.pkl")
    pp = PowerDataset.load_scaler(scaler_path)

    arr = pp.transform(df)
    # use last context_len steps as input
    ctx = ckpt["config"].context_len
    x = torch.from_numpy(arr[-ctx:]).unsqueeze(0).float()

    with torch.no_grad():
        raw_pred = model(x).numpy().flatten()

    predictions = pp.inverse_transform_target(raw_pred).tolist()

    last_ts = pd.Timestamp(df.index[-1])
    timestamps = pd.date_range(start=last_ts, periods=len(predictions)+1, freq="15min")[1:]

    output = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "horizon_steps": len(predictions),
        "forecast": [
            {"timestamp": ts.isoformat(), "predicted_kw": round(float(v), 2)}
            for ts, v in zip(timestamps, predictions)
        ]
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Forecast saved to {args.output}")

if __name__ == "__main__":
    main()
