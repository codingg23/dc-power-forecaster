# dc-power-forecaster

Time-series forecasting for data centre power consumption. Predicts facility-level power draw (kW) up to 24 hours ahead at 15-minute resolution.

Built this as the core forecasting module for Veltora. The goal is to give operators enough advance notice to pre-position cooling, negotiate spot power contracts, and flag anomalies before they cascade.

## Why this matters

Most DCs run reactive cooling - CRAC units respond to temperature after it rises. If you can predict power draw 1-4 hours ahead with reasonable accuracy, you can run cooling proactively and cut PUE meaningfully. Even a 0.05 PUE improvement at a 50MW facility is worth around $300k/year in energy costs.

Forecasting also feeds into the supply chain problem - understanding utilisation trends helps you time capacity additions better.

## Approach

Tried three things, kept the best two.

**LSTM with exogenous features** - works well for 1-6h horizons. Features: time-of-day, day-of-week, recent load history, upcoming calendar events (known downtime windows, planned maintenance).

**Temporal Fusion Transformer (TFT)** - better for 6-24h horizons. The interpretable attention mechanism helps figure out which time features actually matter. Takes longer to train but the uncertainty estimates are genuinely useful.

**Prophet (baseline)** - Facebook's Prophet. Tried it first because it's fast to set up. Decomposition is nice for explainability but the accuracy on DC data is not great. Keeping it as a benchmark.

Multi-step approach is direct multi-output (predicts all horizon steps in one forward pass) rather than recursive. Recursive error compounds too badly at 24h.

## Features

- Sliding window dataset builder from raw sensor CSV/Parquet
- Configurable forecast horizon (15m to 24h)
- LSTM and TFT in PyTorch
- Prophet baseline
- Evaluation suite: MAE, RMSE, MAPE, coverage for probabilistic forecasts
- Anomaly flagging on residuals
- Export predictions as Parquet or JSON

## Tech Stack

- Python 3.11
- PyTorch 2.2
- `pytorch-forecasting` for TFT
- Pandas / NumPy
- Optuna for hyperparameter search
- Weights and Biases for experiment tracking (optional, falls back to local logs)
- Plotly for visualisation

## How to Run

```bash
git clone https://github.com/codingg23/dc-power-forecaster
cd dc-power-forecaster
pip install -r requirements.txt

# train on synthetic data
python train.py --data ./data/power.parquet --model lstm --horizon 96 --epochs 50

# TFT (needs a GPU, much slower)
python train.py --data ./data/power.parquet --model tft --horizon 96 --epochs 30

# evaluate
python evaluate.py --checkpoint ./checkpoints/lstm_best.pt --data ./data/power_test.parquet

# run inference
python predict.py --checkpoint ./checkpoints/lstm_best.pt --data ./data/recent_power.parquet
```

## Results / Learnings

On synthetic data (not a great benchmark, I know):

| Model | MAE (kW) | RMSE (kW) | MAPE | 90% Coverage |
|-------|----------|-----------|------|--------------|
| Prophet | 48.2 | 71.4 | 8.3% | - |
| LSTM | 18.7 | 26.1 | 3.1% | - |
| TFT | 14.3 | 20.8 | 2.4% | 91.2% |

Coverage is for the probabilistic TFT outputs.

Biggest finding: the most important feature by a significant margin is day-of-week combined with hour-of-day. Weekly seasonality in DC load is really strong. Most of what the model is learning is that pattern, plus autocorrelation in the residuals.

Main failure mode: sudden load spikes from batch jobs that weren't in the historical pattern. The model can't predict those but the anomaly detection on residuals catches them quickly.

## Known Issues

- Only trained on synthetic data so far, haven't had a chance to validate on real facility data
- TFT training is slow without a GPU, use LSTM if you're on CPU
- Anomaly flagging is basic (threshold on rolling z-score), needs proper calibration
- Preprocessing pipeline assumes 1-minute input data, needs adjustment for other frequencies

## Next Steps

- Get access to real facility data (talking to a few DC operators)
- Add exogenous signals: outdoor temperature forecast, calendar events
- Proper uncertainty quantification for the LSTM
- Serving layer, right now it's just batch inference
