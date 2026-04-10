import argparse
import io
import os
import pickle
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf


FEATURE_COLS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "ret_1D",
    "ret_3D",
    "ret_7D",
    "rsi",
    "macd",
    "volatility_7D",
    "volume_change",
]

SEQ_LEN = 20
TREND_LABELS = {0: "DOWN", 1: "NEUTRAL", 2: "UP"}

# Default NSE banking universe. Edit this map if you want a different set.
TICKER_MAP = {
    "HDFCBANK": "HDFCBANK.NS",
    "AXISBANK": "AXISBANK.NS",
    "INDUSINDBK": "INDUSINDBK.NS",
    "KOTAKBANK": "KOTAKBANK.NS",
    "SBIN": "SBIN.NS",
    "ICICIBANK": "ICICIBANK.NS",
}


class PortableStandardScaler:
    """Lightweight substitute for sklearn StandardScaler used only for inference."""

    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        self.var_ = None
        self.n_features_in_ = None
        self.n_samples_seen_ = None

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler is missing mean_/scale_ and cannot transform inputs.")
        scale = np.where(np.asarray(self.scale_) == 0, 1.0, np.asarray(self.scale_))
        return (X - np.asarray(self.mean_)) / scale


class _SklearnSafeUnpickler(pickle.Unpickler):
    """Unpickler that remaps sklearn StandardScaler to a local portable class."""

    def find_class(self, module, name):
        if module == "sklearn.preprocessing._data" and name == "StandardScaler":
            return PortableStandardScaler
        return super().find_class(module, name)


def safe_load_artifacts_pickle(path):
    with open(path, "rb") as f:
        data = f.read()

    try:
        return pickle.loads(data)
    except Exception as primary_exc:
        try:
            return _SklearnSafeUnpickler(io.BytesIO(data)).load()
        except Exception as fallback_exc:
            raise RuntimeError(
                f"Failed to load artifacts from {path}. "
                f"Primary error: {primary_exc}. Fallback error: {fallback_exc}"
            ) from fallback_exc


class CNNLSTMDualHead(nn.Module):
    """CNN + BiLSTM dual-head model used in training notebook."""

    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_layers=3,
        dropout=0.2,
        num_classes=3,
        bidirectional=True,
        cnn_channels=64,
    ):
        super().__init__()
        self.num_directions = 2 if bidirectional else 1

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.cnn_branch3 = nn.Sequential(
            nn.Conv1d(hidden_dim, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.GELU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.GELU(),
        )
        self.cnn_branch5 = nn.Sequential(
            nn.Conv1d(hidden_dim, cnn_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels),
            nn.GELU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels),
            nn.GELU(),
        )
        self.cnn_branch7 = nn.Sequential(
            nn.Conv1d(hidden_dim, cnn_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(cnn_channels),
            nn.GELU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(cnn_channels),
            nn.GELU(),
        )

        cnn_out_dim = cnn_channels * 3
        self.cnn_proj = nn.Sequential(
            nn.Conv1d(cnn_out_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.residual_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        lstm_out_dim = hidden_dim * self.num_directions
        self.attention = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        self.shared_mlp = nn.Sequential(
            nn.Linear(lstm_out_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
        )

        self.cls_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

        self.reg_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def attention_pool(self, lstm_out):
        scores = self.attention(lstm_out)
        weights = torch.softmax(scores, dim=1)
        context = (weights * lstm_out).sum(dim=1)
        return context

    def forward(self, x):
        proj = self.input_proj(x)
        proj_t = proj.transpose(1, 2)

        c3 = self.cnn_branch3(proj_t)
        c5 = self.cnn_branch5(proj_t)
        c7 = self.cnn_branch7(proj_t)

        cnn_out = torch.cat([c3, c5, c7], dim=1)
        cnn_out = self.cnn_proj(cnn_out)
        cnn_out = cnn_out.transpose(1, 2)

        gate = self.residual_gate(torch.cat([cnn_out, proj], dim=-1))
        gated = gate * cnn_out + (1 - gate) * proj

        lstm_out, _ = self.lstm(gated)
        context = self.attention_pool(lstm_out)
        shared = self.shared_mlp(context)

        cls_out = self.cls_head(shared)
        reg_out = self.reg_head(shared).squeeze(-1)
        return cls_out, reg_out


def compute_technical_features(price_df):
    dfs = []
    for stock_id, grp in price_df.groupby("stock_id"):
        grp = grp.sort_values("date").copy()
        c = grp["close"]
        v = grp["volume"]

        grp["ret_1D"] = c.pct_change(1)
        grp["ret_3D"] = c.pct_change(3)
        grp["ret_7D"] = c.pct_change(7)
        grp["volatility_7D"] = c.pct_change().rolling(7).std()
        grp["volume_change"] = v.pct_change(1)

        delta = c.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        grp["rsi"] = 100 - (100 / (1 + rs))

        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        grp["macd"] = ema12 - ema26

        dfs.append(grp)

    return pd.concat(dfs, ignore_index=True)


def download_price_data(ticker_map, start_date, end_date):
    tickers = list(ticker_map.values())
    reverse = {v: k for k, v in ticker_map.items()}

    raw = yf.download(
        tickers,
        start=start_date,
        end=(pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        raise RuntimeError("No market data downloaded. Check date/tickers/network.")

    records = []
    if isinstance(raw.columns, pd.MultiIndex):
        for ticker in tickers:
            try:
                df_t = raw.xs(ticker, axis=1, level=1).copy()
            except KeyError:
                continue

            df_t = df_t.rename(columns=str.lower)
            if "close" not in df_t.columns:
                continue

            df_t.index.name = "date"
            df_t = df_t.reset_index()
            df_t["date"] = pd.to_datetime(df_t["date"]).dt.tz_localize(None)
            df_t["stock_id"] = reverse[ticker]
            df_t = df_t[["date", "stock_id", "open", "high", "low", "close", "volume"]]
            df_t = df_t.dropna(subset=["close"])
            records.append(df_t)
    else:
        # Fallback for single-ticker shape
        ticker = tickers[0]
        stock_id = reverse[ticker]
        df_t = raw.rename(columns=str.lower).copy()
        df_t.index.name = "date"
        df_t = df_t.reset_index()
        df_t["date"] = pd.to_datetime(df_t["date"]).dt.tz_localize(None)
        df_t["stock_id"] = stock_id
        df_t = df_t[["date", "stock_id", "open", "high", "low", "close", "volume"]]
        df_t = df_t.dropna(subset=["close"])
        records.append(df_t)

    if not records:
        raise RuntimeError("Could not parse downloaded data for selected tickers.")

    price_df = pd.concat(records, ignore_index=True)
    return price_df.sort_values(["stock_id", "date"]).reset_index(drop=True)


def next_business_day(date_str):
    d = pd.to_datetime(date_str)
    return (d + pd.offsets.BDay(1)).date()


def build_inference_batch(full_data, scaler, asof_date):
    asof = pd.to_datetime(asof_date).tz_localize(None)
    X_batch = []
    meta = []

    for stock_id, grp in full_data.groupby("stock_id"):
        grp = grp.sort_values("date").reset_index(drop=True)
        grp = grp[grp["date"] <= asof].copy()
        if grp.empty:
            continue

        grp[FEATURE_COLS] = grp[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
        grp[FEATURE_COLS] = grp[FEATURE_COLS].ffill().bfill()

        anchor_idx = len(grp) - 1
        if anchor_idx < SEQ_LEN:
            continue

        feats = scaler.transform(grp[FEATURE_COLS].values)
        seq = feats[anchor_idx - SEQ_LEN : anchor_idx]

        X_batch.append(seq)
        meta.append(
            {
                "stock_id": stock_id,
                "anchor_date": grp["date"].iloc[anchor_idx].date(),
                "last_close": float(grp["close"].iloc[anchor_idx]),
            }
        )

    if not X_batch:
        raise RuntimeError(
            "No inference samples available. Try a later date with enough history."
        )

    X = torch.tensor(np.array(X_batch), dtype=torch.float32)
    return X, meta


def load_artifacts(base_dir, device):
    model_path = os.path.join(base_dir, "saved_model_cnn", "dual_head_transformer.pt")
    scaler_path = os.path.join(base_dir, "saved_model_cnn", "scaler.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

    artifacts = safe_load_artifacts_pickle(scaler_path)

    scaler = artifacts["scaler"]
    yr_mean = float(artifacts["yr_mean"])
    yr_std = float(artifacts["yr_std"])

    model = CNNLSTMDualHead(
        input_dim=len(FEATURE_COLS),
        hidden_dim=128,
        num_layers=3,
        cnn_channels=64,
        dropout=0.2,
        bidirectional=True,
    ).to(device)

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    return model, scaler, yr_mean, yr_std


def run_prediction(asof_date):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    preferred_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    active_device = preferred_device

    try:
        model, scaler, yr_mean, yr_std = load_artifacts(base_dir, active_device)
    except Exception as exc:
        if preferred_device.type == "cuda":
            msg = str(exc).lower()
            if "cuda" in msg or "accelerator" in msg or "cudnn" in msg:
                print("CUDA model load failed. Falling back to CPU...")
                active_device = torch.device("cpu")
                model, scaler, yr_mean, yr_std = load_artifacts(base_dir, active_device)
            else:
                raise
        else:
            raise

    start_date = (pd.to_datetime(asof_date) - timedelta(days=400)).strftime("%Y-%m-%d")
    price_df = download_price_data(TICKER_MAP, start_date=start_date, end_date=asof_date)
    full_data = compute_technical_features(price_df)

    X, meta = build_inference_batch(full_data, scaler=scaler, asof_date=asof_date)

    try:
        X_device = X.to(active_device)
        with torch.no_grad():
            cls_logits, reg_pred = model(X_device)
    except Exception as exc:
        if active_device.type == "cuda":
            msg = str(exc).lower()
            if "cuda" in msg or "accelerator" in msg:
                print("CUDA inference failed. Retrying on CPU...")
                model, scaler, yr_mean, yr_std = load_artifacts(base_dir, torch.device("cpu"))
                X_cpu, meta = build_inference_batch(full_data, scaler=scaler, asof_date=asof_date)
                with torch.no_grad():
                    cls_logits, reg_pred = model(X_cpu)
            else:
                raise
        else:
            raise

    probs = torch.softmax(cls_logits, dim=-1).cpu().numpy()
    cls_idx = cls_logits.argmax(dim=1).cpu().numpy()
    pred_close = (reg_pred.cpu().numpy() * yr_std) + yr_mean

    pred_date = next_business_day(asof_date)
    rows = []
    for i, item in enumerate(meta):
        direction = TREND_LABELS[int(cls_idx[i])]
        confidence = float(probs[i][int(cls_idx[i])])

        rows.append(
            {
                "asof_date": str(item["anchor_date"]),
                "predicted_for": str(pred_date),
                "stock_id": item["stock_id"],
                "last_close": round(item["last_close"], 2),
                "pred_next_close": round(float(pred_close[i]), 2),
                "pred_movement": direction,
                "confidence": round(confidence, 4),
                "p_down": round(float(probs[i][0]), 4),
                "p_neutral": round(float(probs[i][1]), 4),
                "p_up": round(float(probs[i][2]), 4),
            }
        )

    out = pd.DataFrame(rows).sort_values("stock_id").reset_index(drop=True)

    out_dir = os.path.join(base_dir, "saved_model", "predictions")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"predictions_{asof_date}.csv")
    out.to_csv(out_file, index=False)

    print("\nPrediction results:")
    print(out.to_string(index=False))
    print(f"\nSaved CSV: {out_file}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict next-day price and movement from saved model using only date input."
    )
    parser.add_argument(
        "--date",
        type=str,
        help="As-of date in YYYY-MM-DD format (for example 2026-03-28).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    asof_date = args.date

    if not asof_date:
        asof_date = input("Enter date (YYYY-MM-DD): ").strip()

    try:
        pd.to_datetime(asof_date)
    except Exception as exc:
        raise ValueError("Invalid date format. Use YYYY-MM-DD.") from exc

    run_prediction(asof_date)


if __name__ == "__main__":
    main()
