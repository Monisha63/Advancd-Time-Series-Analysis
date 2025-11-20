#!/usr/bin/env python3
"""
Advanced Time Series Forecasting with Transformer + Attention
All outputs are saved to Desktop:
- attention.png
- report.txt
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

# statsmodels
import statsmodels.api as sm

# yfinance
import yfinance as yf

# --------------------------------------------------------------------
# Output folder (force save to Desktop)
# --------------------------------------------------------------------
OUTPUT_DIR = r"C:/Users/monis/Desktop/"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print("Saving all output files to:", OUTPUT_DIR)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# --------------------------------------------------------------------
# Utility Functions
# --------------------------------------------------------------------
def rmse(y, p):
    return math.sqrt(mean_squared_error(y, p))

def mape(y, p):
    y = np.array(y)
    p = np.array(p)
    return np.mean(np.abs((y - p) / np.maximum(np.abs(y), 1e-8))) * 100.0

def load_sp500(start="2015-01-01", end=None):
    df = yf.download("^GSPC", start=start, end=end, progress=False)

    # yfinance now returns adjusted data in "Close"
    if "Adj Close" in df.columns:
        df = df[["Adj Close"]].rename(columns={"Adj Close": "adj_close"})
    elif "Close" in df.columns:
        df = df[["Close"]].rename(columns={"Close": "adj_close"})
    else:
        raise KeyError("No usable price column found!")

    return df.dropna()

def engineer_features(df):
    df = df.copy()
    df["ret"] = df["adj_close"].pct_change().fillna(0)
    df["ma_5"] = df["adj_close"].rolling(5).mean().bfill()
    df["ma_21"] = df["adj_close"].rolling(21).mean().bfill()
    df["vol_10"] = df["ret"].rolling(10).std().bfill()
    return df.dropna()

def make_sequences(values, seq_len=60, horizon=5):
    X, Y = [], []
    T = len(values)
    for i in range(T - seq_len - horizon):
        X.append(values[i:i + seq_len])
        Y.append(values[i + seq_len:i + seq_len + horizon, 0])
    return np.array(X), np.array(Y)

class TimeSeriesDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.Y[i]

# --------------------------------------------------------------------
# Transformer Model
# --------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerModel(nn.Module):
    def __init__(self, n_features, d_model=64, n_heads=4, n_layers=2, dim_ff=128, horizon=5):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.pos = PositionalEncoding(d_model)
        self.layers = nn.ModuleList()

        for _ in range(n_layers):
            attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            ff = nn.Sequential(nn.Linear(d_model, dim_ff), nn.GELU(), nn.Linear(dim_ff, d_model))
            ln1 = nn.LayerNorm(d_model)
            ln2 = nn.LayerNorm(d_model)
            self.layers.append(nn.ModuleDict({"attn": attn, "ff": ff, "ln1": ln1, "ln2": ln2}))

        self.head = nn.Linear(d_model, horizon)

    def forward(self, x, return_attn=False):
        x = self.pos(self.proj(x))
        att_out = []

        for layer in self.layers:
            att, w = layer["attn"](x, x, x, need_weights=True)
            x = layer["ln1"](x + att)
            x = layer["ln2"](x + layer["ff"](x))
            att_out.append(w)

        pooled = x.mean(dim=1)
        preds = self.head(pooled)

        if return_attn:
            return preds, att_out
        return preds

# --------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------
def sarima_baseline(series, steps):
    try:
        model = sm.tsa.statespace.SARIMAX(series, order=(1,1,1))
        res = model.fit(disp=False)
        return res.forecast(steps)
    except:
        return np.repeat(series[-1], steps)

class DenseBaseline(nn.Module):
    def __init__(self, input_size, horizon):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, horizon)
        )
    def forward(self, x):
        return self.nn(x)

# --------------------------------------------------------------------
# Training Helpers
# --------------------------------------------------------------------
def train_epoch(model, loader, opt, loss):
    model.train()
    total = 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        pred = model(xb)
        l = loss(pred, yb)
        opt.zero_grad()
        l.backward()
        opt.step()
        total += l.item() * len(xb)
    return total / len(loader.dataset)

def evaluate(model, loader, loss):
    model.eval()
    preds, trues = [], []
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            p = model(xb)
            l = loss(p, yb)
            preds.append(p.cpu().numpy())
            trues.append(yb.cpu().numpy())
            total += l.item() * len(xb)
    return total / len(loader.dataset), np.vstack(trues), np.vstack(preds)

# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------
def main():

    # 1. Load data
    df = load_sp500("2015-01-01")
    df = engineer_features(df)

    scaler = StandardScaler()
    vals = scaler.fit_transform(df.values)

    seq_len = 60
    horizon = 5

    X, Y = make_sequences(vals, seq_len, horizon)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]

    train_dl = DataLoader(TimeSeriesDataset(X_train, Y_train), batch_size=64, shuffle=True)
    test_dl  = DataLoader(TimeSeriesDataset(X_test, Y_test), batch_size=64)

    n_features = X.shape[2]

    # 2. Model
    model = TransformerModel(n_features=n_features).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # 3. Training
    print("\nTraining Transformer...")
    for e in range(10):
        tr = train_epoch(model, train_dl, opt, loss_fn)
        te, _, _ = evaluate(model, test_dl, loss_fn)
        print(f"Epoch {e+1}/10 → train={tr:.4f}, test={te:.4f}")

    # 4. Final evaluation
    _, y_scaled, p_scaled = evaluate(model, test_dl, loss_fn)

    def invert(arr):
        out = []
        for row in arr:
            new = []
            for v in row:
                temp = np.zeros(vals.shape[1])
                temp[0] = v
                new.append(scaler.inverse_transform([temp])[0][0])
            out.append(new)
        return np.array(out)

    y_true = invert(y_scaled)
    y_pred = invert(p_scaled)

    mae = mean_absolute_error(y_true[:,0], y_pred[:,0])
    r = rmse(y_true[:,0], y_pred[:,0])
    m = mape(y_true[:,0], y_pred[:,0])

    print("\nTransformer Performance:")
    print("MAE:", mae)
    print("RMSE:", r)
    print("MAPE:", m)

    # 5. Attention heatmap
    print("\nSaving attention.png ...")
    sample_x = torch.tensor(X_test[0:1], dtype=torch.float32).to(DEVICE)
    _, att = model(sample_x, return_attn=True)
    last_attn = att[-1][0].detach().cpu().numpy()  # FIXED

    plt.imshow(last_attn, cmap="viridis")
    plt.colorbar()
    plt.title("Attention Heatmap")
    plt.savefig(OUTPUT_DIR + "attention.png")
    plt.close()

    # 6. Report
    print("Saving report.txt ...")
    with open(OUTPUT_DIR + "report.txt", "w") as f:
        f.write("Advanced Time Series Forecasting Report\n")
        f.write(f"Generated: {datetime.now()}\n\n")
        f.write("Transformer Model Results:\n")
        f.write(f"MAE: {mae:.4f}\nRMSE: {r:.4f}\nMAPE: {m:.2f}%\n")

    print("\n✔ All files saved to Desktop!")
    print(" - attention.png")
    print(" - report.txt")

if __name__ == "__main__":
    main()
