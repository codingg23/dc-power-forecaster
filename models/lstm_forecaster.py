"""
lstm_forecaster.py

LSTM-based multi-step power forecaster.

Architecture: encoder-decoder LSTM with attention.
- Encoder: processes historical context window
- Decoder: auto-regressive or direct multi-output

Went with direct multi-output for the main use case (24h horizon)
because recursive error compound is brutal over 96 steps.
The auto-regressive mode is still there for short horizons where
you want to condition on intermediate predictions.

Input features per timestep:
    - power_kw (target, lagged)
    - hour_sin, hour_cos (cyclic encoding)
    - dow_sin, dow_cos  (cyclic day of week)
    - is_weekend
    - recent_delta (first difference — helps the model see rate of change)

Could add more exogenous features (outdoor temp, calendar events) but
the above already gets you most of the way there. YAGNI for now.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class LSTMConfig:
    input_size: int = 7           # number of input features
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.15
    horizon: int = 96             # forecast steps (96 * 15min = 24h)
    context_len: int = 672        # 7 days of 15-min data
    direct_output: bool = True    # False = autoregressive
    bidirectional_encoder: bool = False  # tried it, didn't help much


class AttentionLayer(nn.Module):
    """
    Simple additive attention over encoder hidden states.
    Helps the decoder focus on relevant parts of the context window.
    Not multi-head — kept it simple since the performance difference
    on this dataset didn't justify the extra complexity.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.query_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """
        query: (batch, hidden)
        keys:  (batch, seq_len, hidden)
        returns: (batch, hidden) — context vector
        """
        q = self.query_proj(query).unsqueeze(1)  # (batch, 1, hidden)
        k = self.key_proj(keys)                   # (batch, seq_len, hidden)
        scores = self.v_proj(torch.tanh(q + k)).squeeze(-1)  # (batch, seq_len)
        weights = F.softmax(scores, dim=-1)
        context = (weights.unsqueeze(-1) * keys).sum(dim=1)  # (batch, hidden)
        return context


class LSTMForecaster(nn.Module):

    def __init__(self, config: LSTMConfig):
        super().__init__()
        self.config = config

        encoder_output_size = (
            config.hidden_size * 2 if config.bidirectional_encoder else config.hidden_size
        )

        self.encoder = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=config.bidirectional_encoder,
        )

        self.attention = AttentionLayer(encoder_output_size)

        if config.direct_output:
            # project directly to all forecast steps
            self.output_head = nn.Sequential(
                nn.Linear(encoder_output_size, 256),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(256, config.horizon),
            )
        else:
            # decoder LSTM for autoregressive mode
            self.decoder_cell = nn.LSTMCell(
                input_size=1 + encoder_output_size,
                hidden_size=config.hidden_size,
            )
            self.output_proj = nn.Linear(config.hidden_size, 1)

        self.layer_norm = nn.LayerNorm(encoder_output_size)

    def forward(
        self,
        x: torch.Tensor,
        teacher_forcing_ratio: float = 0.0,
        targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: (batch, context_len, input_size)
        targets: (batch, horizon) — only used during training for teacher forcing
        returns: (batch, horizon)
        """
        batch_size = x.size(0)

        encoder_out, (h_n, c_n) = self.encoder(x)
        # encoder_out: (batch, context_len, hidden * dirs)

        # use last hidden state as query for attention
        if self.config.bidirectional_encoder:
            last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            last_hidden = h_n[-1]

        context = self.attention(last_hidden, encoder_out)
        context = self.layer_norm(context)

        if self.config.direct_output:
            return self.output_head(context)

        # autoregressive decoding
        # initialize decoder state from encoder
        # only use the top layer, non-bidirectional decoder
        h = h_n[-1] if not self.config.bidirectional_encoder else last_hidden[:, :self.config.hidden_size]
        c = c_n[-1] if not self.config.bidirectional_encoder else torch.zeros_like(h)

        # seed with last observed value
        last_obs = x[:, -1, 0:1]  # (batch, 1) — the power_kw feature
        outputs = []

        for step in range(self.config.horizon):
            dec_input = torch.cat([last_obs, context], dim=-1)
            h, c = self.decoder_cell(dec_input, (h, c))
            pred = self.output_proj(h)
            outputs.append(pred)

            # teacher forcing: sometimes feed ground truth instead of prediction
            if targets is not None and np.random.random() < teacher_forcing_ratio:
                last_obs = targets[:, step:step + 1]
            else:
                last_obs = pred.detach()

        return torch.cat(outputs, dim=-1)  # (batch, horizon)


def build_model(config: Optional[LSTMConfig] = None) -> LSTMForecaster:
    if config is None:
        config = LSTMConfig()
    return LSTMForecaster(config)


if __name__ == "__main__":
    # quick sanity check
    config = LSTMConfig(context_len=96, horizon=24)
    model = build_model(config)
    x = torch.randn(8, config.context_len, config.input_size)
    out = model(x)
    print(f"Output shape: {out.shape}")  # expect (8, 24)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")
