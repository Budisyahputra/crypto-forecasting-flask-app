import os
os.environ["KERAS_BACKEND"] = "tensorflow"

from flask import Flask, request, render_template
import json
import numpy as np
import pandas as pd
import yfinance as yf

import keras
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

TICKERS = {"BTC": "BTC-USD", "ETH": "ETH-USD", "BNB": "BNB-USD"}

MODEL_DIRS = {
    "BTC": "models_fixed/BTC_savedmodel",
    "ETH": "models_fixed/ETH_savedmodel",
    "BNB": "models_fixed/BNB_savedmodel",
}

NPZ_SCALER_PATHS = {
    "BTC": "scalers_npz/BTC-USD_H1_targetClose.npz",
    "ETH": "scalers_npz/ETH-USD_H1_targetClose.npz",
    "BNB": "scalers_npz/BNB-USD_H1_targetClose.npz",
}

CONFIG_PATH = "config.json"

MODELS = {}
SCALERS = {}

# ---------- Utils ----------
def get_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def fetch_close(asset: str, period: str = "2y", interval: str = "1d") -> pd.Series:
    df = yf.download(
        TICKERS[asset],
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )
    if df is None or df.empty:
        raise ValueError(f"Data Yahoo Finance kosong untuk {asset}.")
    df = df.dropna()
    if "Close" not in df.columns:
        raise ValueError("Kolom Close tidak ditemukan.")
    return df["Close"]

def make_input_window(close_series: pd.Series, lookback: int) -> np.ndarray:
    if len(close_series) < lookback:
        raise ValueError(f"Data kurang: butuh >= {lookback}, dapat {len(close_series)}.")
    return close_series.iloc[-lookback:].to_numpy().reshape(-1, 1)

def load_scaler_from_npz(npz_path: str) -> StandardScaler:
    z = np.load(npz_path, allow_pickle=True)
    scaler = StandardScaler()
    scaler.mean_ = z["scaler_mean"]
    scaler.scale_ = z["scaler_scale"]
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = scaler.mean_.shape[0]
    return scaler

def get_scaler(asset: str) -> StandardScaler:
    if asset not in SCALERS:
        SCALERS[asset] = load_scaler_from_npz(NPZ_SCALER_PATHS[asset])
    return SCALERS[asset]

def build_tfsm_model(savedmodel_dir: str, lookback: int):
    layer = keras.layers.TFSMLayer(savedmodel_dir, call_endpoint="serving_default")
    inp = keras.Input(shape=(lookback, 1))
    out = layer(inp)
    if isinstance(out, dict):
        out = list(out.values())[0]
    return keras.Model(inp, out)

def get_model(asset: str, lookback: int):
    if asset not in MODELS:
        MODELS[asset] = build_tfsm_model(MODEL_DIRS[asset], lookback)
    return MODELS[asset]

def fmt_money(x): return f"{x:,.2f}"
def fmt_pct(x): return f"{x:.2f}%"

def predict_h1(asset: str):
    lookback = int(get_config().get("lookback", 60))
    close = fetch_close(asset)
    window = make_input_window(close, lookback)

    last_close = float(close.iloc[-1])
    last_date = close.index[-1].date()
    target_date = last_date + pd.Timedelta(days=1)

    scaler = get_scaler(asset)
    model = get_model(asset, lookback)

    X = scaler.transform(window).reshape(1, lookback, 1)
    y = model.predict(X, verbose=0).reshape(-1, 1)
    y_pred = float(scaler.inverse_transform(y)[0, 0])

    delta = y_pred - last_close
    pct = (delta / last_close) * 100

    return {
        "asset": asset,
        "last_close": last_close,
        "pred_h1": y_pred,
        "delta": delta,
        "pct": pct,
        "trend": "up" if delta >= 0 else "down",
        "lookback": lookback,
        "last_date": str(last_date),
        "target_date": str(target_date),
        "note": "Scaler: StandardScaler (NPZ, fit train). Model: SavedModel (TFSMLayer)."
    }

# ---------- Routes ----------
@app.route("/", methods=["GET", "POST"])
def index():
    asset = request.form.get("asset", "BTC")
    result, error = None, None

    if request.method == "POST":
        try:
            result = predict_h1(asset)
        except Exception as e:
            error = str(e)

    return render_template(
        "index.html",
        asset=asset,
        result=result,
        error=error,
        fmt_money=fmt_money,
        fmt_pct=fmt_pct,
        lookback=get_config().get("lookback", 60)
    )

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
