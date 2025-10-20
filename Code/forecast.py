# forecast.py
import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from seed_utils import set_all_seeds

# ───────────────────────── CONFIG ─────────────────────────
PROFILE     = os.environ.get("PROFILE", "ceemd_cnnlstm_rl_ta_macro")
TICKERS     = os.environ.get("TICKERS", "AAPL,JPM,AMZN,TSLA,MSFT").split(",")
SEQ_LEN     = int(os.environ.get("SEQ_LEN", "60"))
TRAIN_END_Y = int(os.environ.get("TRAIN_END_Y", "2022"))
SCALER_TYPE = os.environ.get("SCALER_TYPE", "minmax")
MODEL_TYPE  = os.environ.get("MODEL_TYPE", "cnn_lstm")   
INCLUDE_PRED_CHANNEL = os.environ.get("INCLUDE_PRED_CHANNEL", "1") == "1"
SEED        = int(os.environ.get("SEED", "42"))

RUN_DIR     = os.path.join("runs", PROFILE)
IN_DIR      = os.path.join(RUN_DIR, "preprocessed")
OUT_DIR     = os.path.join(RUN_DIR, "forecast")
FIG_DIR     = os.path.join(OUT_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
set_all_seeds(SEED)

PROFILES_FOR_FIGS = {
    "CEEMD–CNN–LSTM + Macro": "ceemd_cnnlstm_rl_ta_macro",
    "CEEMD–CNN–LSTM":         "ceemd_cnnlstm_rl_ta",
    "CNN–LSTM":               "cnnlstm_rl_ta",
    "LSTM":                   "lstm_rl_ta",
}

def create_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len]); ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)

def get_scaler(kind="minmax"):
    return MinMaxScaler() if kind == "minmax" else StandardScaler()

def build_model(input_shape, kind="cnn_lstm"):
    m = Sequential()
    if kind == "lstm_only":
        m.add(LSTM(64, input_shape=input_shape))
        m.add(Dropout(0.2)); m.add(Dense(1))
    else:
        m.add(Conv1D(32, 3, activation='relu', padding='same', input_shape=input_shape))
        m.add(MaxPooling1D(2)); m.add(LSTM(64)); m.add(Dropout(0.2)); m.add(Dense(1))
    m.compile(optimizer=Adam(1e-4), loss='mse', metrics=['mae'])
    return m

def directional_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    prev = np.roll(y_true, 1)
    mask = np.ones_like(y_true, dtype=bool); mask[0] = False
    s_true = np.sign(y_true[mask] - prev[mask])
    s_pred = np.sign(y_pred[mask] - prev[mask])
    return float((s_true == s_pred).mean()) if s_true.size else float("nan")

# ──────────────────────── TRAIN & SAVE ────────────────────────
if __name__ == "__main__":
    blocks, feat_dims = [], []
    for t in TICKERS:
        path = os.path.join(IN_DIR, t)
        X_raw = np.load(os.path.join(path, "X_raw.npy"))
        y_raw = np.load(os.path.join(path, "y_raw.npy"))
        df_ta = pd.read_csv(os.path.join(path, "ta.csv"), index_col=0, parse_dates=True)
        dates = df_ta.index
        feat_dims.append(X_raw.shape[1])
        blocks.append((t, X_raw, y_raw, dates))
    max_dim = max(feat_dims)

    
    Xtr_concat, ytr_concat = [], []
    train_masks_per_t = {}
    for t, X_raw, y_raw, dates in blocks:
        if X_raw.shape[1] < max_dim:
            pad = np.zeros((X_raw.shape[0], max_dim - X_raw.shape[1]), dtype=X_raw.dtype)
            X_raw = np.hstack([X_raw, pad])
        train_mask = dates.year <= TRAIN_END_Y
        train_masks_per_t[t] = train_mask
        Xtr_concat.append(X_raw[train_mask])
        ytr_concat.append(y_raw[train_mask])
    Xtr_concat = np.vstack([x for x in Xtr_concat if len(x)])
    ytr_concat = np.hstack([y for y in ytr_concat if len(y)])

    scalerX_global = get_scaler(SCALER_TYPE).fit(Xtr_concat)
    scalerY_global = MinMaxScaler().fit(ytr_concat.reshape(-1,1))

    all_Xtr, all_ytr, all_Xte, all_yte, all_te_dates = [], [], [], [], []
    per_t_transformed = {}

    for t, X_raw, y_raw, dates in blocks:
        if X_raw.shape[1] < max_dim:
            pad = np.zeros((X_raw.shape[0], max_dim - X_raw.shape[1]), dtype=X_raw.dtype)
            X_raw = np.hstack([X_raw, pad])

        train_mask = train_masks_per_t[t]
        test_mask  = dates.year >= (TRAIN_END_Y + 1)

        X = scalerX_global.transform(X_raw)
        y = scalerY_global.transform(y_raw.reshape(-1,1)).flatten()

        Xseq, yseq = create_sequences(X, y, SEQ_LEN)
        seq_dates  = dates[SEQ_LEN:]
        tr_mask, te_mask = train_mask[SEQ_LEN:], test_mask[SEQ_LEN:]

        all_Xtr.append(Xseq[tr_mask]); all_ytr.append(yseq[tr_mask])
        all_Xte.append(Xseq[te_mask]);  all_yte.append(yseq[te_mask])
        all_te_dates.append(seq_dates[te_mask])

        per_t_transformed[t] = {
            "Xseq": Xseq, "yseq": yseq, "dates_seq": seq_dates,
            "tr_mask": tr_mask, "te_mask": te_mask
        }

    X_train = np.vstack(all_Xtr)
    y_train = np.hstack(all_ytr)
    X_test  = np.vstack(all_Xte) if any(len(x) for x in all_Xte) else np.empty((0, SEQ_LEN, max_dim))
    y_test  = np.hstack(all_yte) if any(len(y) for y in all_yte) else np.empty((0,))
    te_dates= np.concatenate(all_te_dates) if any(len(d) for d in all_te_dates) else np.array([], dtype="datetime64[ns]")

    model = build_model((SEQ_LEN, max_dim), MODEL_TYPE)
    cbs = [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6),
        ModelCheckpoint(os.path.join(OUT_DIR, "best.h5"), monitor='val_loss', save_best_only=True)
    ]
    hist = model.fit(X_train, y_train, validation_split=0.1, epochs=60, batch_size=32, callbacks=cbs, verbose=0)
    model.save(os.path.join(OUT_DIR, "final.h5"))

    pooled_metrics = {}
    if len(X_test):
        y_pred = model.predict(X_test, verbose=0).flatten()
        pooled_metrics["rmse"] = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        pooled_metrics["mae"]  = float(mean_absolute_error(y_test, y_pred))
        pooled_metrics["da"]   = float(directional_accuracy(y_test, y_pred))
    else:
        y_pred = np.array([], dtype=np.float32)
        pooled_metrics = {"rmse": None, "mae": None, "da": None}

    with open(os.path.join(OUT_DIR, "pooled_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(pooled_metrics, f, indent=2)

    start = 0
    for t, X_raw, y_raw, dates in blocks:
        pt = per_t_transformed[t]
        Xseq, yseq, dates_seq = pt["Xseq"], pt["yseq"], pt["dates_seq"]
        tr_mask, te_mask = pt["tr_mask"], pt["te_mask"]

        n_te = te_mask.sum()
        if n_te > 0 and len(X_test):
            y_pred_part = y_pred[start:start+n_te]
        else:
            y_pred_part = np.zeros((n_te,), dtype=np.float32)
        start += n_te

        if n_te > 0:
            m = {
                "rmse": float(np.sqrt(mean_squared_error(yseq[te_mask], y_pred_part))),
                "mae":  float(mean_absolute_error(yseq[te_mask], y_pred_part)),
                "da":   float(directional_accuracy(yseq[te_mask], y_pred_part)),
                "n":    int(n_te)
            }
        else:
            m = {"rmse": None, "mae": None, "da": None, "n": 0}

        def make_state(xseq, yhat=None):
            base = xseq[:, -1, :]
            return np.hstack([base, yhat.reshape(-1,1)]) if (INCLUDE_PRED_CHANNEL and yhat is not None and len(yhat)==len(base)) else base

        X_base_tr   = Xseq[tr_mask]
        state_train = make_state(X_base_tr, None)

        X_base_te   = Xseq[te_mask]
        state_test  = make_state(X_base_te, y_pred_part if INCLUDE_PRED_CHANNEL else None)

        outp = os.path.join(OUT_DIR, t); os.makedirs(outp, exist_ok=True)
        np.save(os.path.join(outp, "state_train.npy"), state_train.astype(np.float32))
        np.save(os.path.join(outp, "dates_train.npy"), dates_seq[tr_mask].to_numpy())
        np.save(os.path.join(outp, "state_test.npy"),  state_test.astype(np.float32))
        np.save(os.path.join(outp, "dates_test.npy"),  dates_seq[te_mask].to_numpy())

        with open(os.path.join(outp, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(m, f, indent=2)

        if n_te > 0:
            y_true_te = scalerY_global.inverse_transform(yseq[te_mask].reshape(-1, 1)).flatten()
            if len(X_test):
                y_pred_te = scalerY_global.inverse_transform(y_pred_part.reshape(-1, 1)).flatten()
            else:
                y_pred_te = np.array([], dtype=np.float32)
            df_fore = pd.DataFrame({
                "date": pd.to_datetime(dates_seq[te_mask]),
                "actual_price": y_true_te.astype(float),
                "pred_price":   y_pred_te.astype(float)
            }).sort_values("date")
            df_fore.to_csv(os.path.join(outp, "forecast_test.csv"), index=False)

    print(f"✅ Forecasting done -> {OUT_DIR} | pooled_metrics={pooled_metrics}")

    