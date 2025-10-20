# preprocess.py

import os, json
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import numpy as np
import pandas as pd
import yaml
import yfinance as yf

try:
    from pandas_datareader import data as pdr
except Exception as _pdr_err:
    pdr = None
    _PDR_IMPORT_ERROR = _pdr_err

try:
    from PyEMD import CEEMDAN
except Exception as _ceemdan_err:
    CEEMDAN = None
    _CEEMDAN_IMPORT_ERROR = _ceemdan_err

try:
    from seed_utils import set_all_seeds
except Exception:
    def set_all_seeds(seed: int, deterministic_torch: bool = True):
        np.random.seed(int(seed))

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _md5_bytes(b: bytes) -> str:
    import hashlib
    m = hashlib.md5()
    m.update(b)
    return m.hexdigest()

def _md5_df(df: pd.DataFrame) -> str:
    csv = df.to_csv(None, float_format="%.10g").encode("utf-8")
    return _md5_bytes(csv)

def _to_parquet_or_csv(df: pd.DataFrame, path_parquet: Path):
    try:
        df.to_parquet(path_parquet)
        return "parquet"
    except Exception:
        df.to_csv(path_parquet.with_suffix(".csv"))
        return "csv"

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
def _find_cfg_path() -> Path:
    env_p = os.environ.get("EXP_YAML", "").strip()
    if env_p:
        p = Path(env_p).expanduser().resolve()
        if p.exists():
            return p
        raise FileNotFoundError(f"EXP_YAML set but file not found: {p}")
    for cand in (Path("exp.yaml"), Path("configs/exp.yaml")):
        if cand.exists():
            return cand.resolve()
    raise FileNotFoundError("Could not find exp.yaml in ./ or ./configs/ (or via EXP_YAML).")

def _get_cfg():
    cfg_path = _find_cfg_path()
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    profiles = cfg["profiles"]
    profile = os.environ.get("PROFILE", list(profiles.keys())[0])
    pconf = profiles[profile]
    g = cfg["global"]

    tickers = os.environ.get("TICKERS", ",".join(g["tickers"])).split(",")
    start   = os.environ.get("START", g["start"])
    end     = os.environ.get("END",   g["end"])
    train_end_y = int(os.environ.get("TRAIN_END_Y", g.get("train_end_y", 2022)))
    seq_len = int(os.environ.get("SEQ_LEN", g.get("seq_len", 60)))
    scaler_type = os.environ.get("SCALER_TYPE", g.get("scaler_type", "minmax"))
    seed    = int(os.environ.get("SEED", g.get("seed", 42)))

    use_ceemdan    = int(os.environ.get("USE_CEEMDAN", 1 if pconf.get("use_ceemdan", 0) else 0))
    include_macro  = int(os.environ.get("INCLUDE_MACRO", 1 if pconf.get("include_macro", 0) else 0))

    ceemdan_params = g.get("ceemdan", {})
    ceemdan_params.update({
        "max_imf":     int(os.environ.get("MAX_IMF",     ceemdan_params.get("max_imf", 6))),
        "trials":      int(os.environ.get("TRIALS",      ceemdan_params.get("trials", 100))),
        "noise_width": float(os.environ.get("NOISE_WIDTH", ceemdan_params.get("noise_width", 0.05))),
        "energy_min":  float(os.environ.get("ENERGY_MIN",  ceemdan_params.get("energy_min", 0.02))),
        "corr_min":    float(os.environ.get("CORR_MIN",    ceemdan_params.get("corr_min", 0.10))),
    })

    return {
        "profile": profile,
        "tickers": tickers,
        "start": start,
        "end": end,
        "train_end_y": train_end_y,
        "seq_len": seq_len,
        "scaler_type": scaler_type,
        "seed": seed,
        "use_ceemdan": use_ceemdan,
        "include_macro": include_macro,
        "ceemdan": ceemdan_params,
    }

# ──────────────────────────────────────────────────────────────────────────────
# Macro Creation and Alignment
# ──────────────────────────────────────────────────────────────────────────────
def ensure_macro_monthly(csv_path: Path, start: str, end: str,
                         lag_months: Dict[str, int] | None = None):
    """
    Create/refresh macro monthly csv with coarse release-aware lags to reduce leakage.
      - gap_growth (Real GDP vs Potential GDP): (GDPC1 - GDPPOT)/GDPPOT  [Q→M ffill, lag 1M]
      - inflation YoY% (CPIAUCSL)                                          [M,  lag 1M]
      - unemployment (UNRATE %)                                            [M,  lag 1M]
      - vix (VIXCLS, monthly ffill of daily)                               [D→M ffill, lag 0M]
    """
    if pdr is None:
        raise ImportError(
            "Missing dependency: pandas-datareader. "
            "Install with: pip install pandas-datareader"
        )

    if lag_months is None:
        lag_months = {"gap_growth": 1, "inflation": 1, "unemployment": 1, "vix": 0}

    _ensure_dir(csv_path.parent)

    start_dt, end_dt = pd.to_datetime(start), pd.to_datetime(end)

    gdp = pdr.DataReader("GDPC1",   "fred", start_dt, end_dt)  
    pot = pdr.DataReader("GDPPOT",  "fred", start_dt, end_dt)  
    cpi = pdr.DataReader("CPIAUCSL","fred", start_dt, end_dt)  
    un  = pdr.DataReader("UNRATE",  "fred", start_dt, end_dt)  
    vix = pdr.DataReader("VIXCLS",  "fred", start_dt, end_dt)  

    gdp_q = gdp.asfreq("QE").ffill()
    pot_q = pot.asfreq("QE").ffill()
    gap = (gdp_q["GDPC1"] - pot_q["GDPPOT"]) / pot_q["GDPPOT"]
    gap.name = "gap_growth"
    gap_m = gap.resample("MS").ffill()

    cpi_m = cpi.asfreq("MS").ffill()
    infl = 100.0 * (cpi_m["CPIAUCSL"] / cpi_m["CPIAUCSL"].shift(12) - 1.0)
    infl.name = "inflation"

    un_m = un.asfreq("MS").ffill()
    unemp = un_m["UNRATE"]; unemp.name = "unemployment"

    vix_m = vix.asfreq("MS").ffill()
    vix_m = vix_m["VIXCLS"]; vix_m.name = "vix"

    monthly = pd.concat([gap_m, infl, unemp, vix_m], axis=1).sort_index()

    for k, lag in lag_months.items():
        if k in monthly.columns and lag:
            monthly[k] = monthly[k].shift(lag)

    monthly = monthly.dropna(how="all")
    monthly.index.name = "date"
    monthly.to_csv(csv_path)
    print(f"[preprocess] Created/updated macro monthly file -> {csv_path} (rows={len(monthly)})")

def macro_daily_from_monthly(monthly_csv: Path, daily_index: pd.DatetimeIndex) -> pd.DataFrame:
    m = pd.read_csv(monthly_csv, parse_dates=["date"]).set_index("date").sort_index()
    df = m.reindex(pd.date_range(m.index.min(), daily_index.max(), freq="B")).ffill()
    df = df.reindex(daily_index).ffill()
    return df

# ──────────────────────────────────────────────────────────────────────────────
# Prices
# ──────────────────────────────────────────────────────────────────────────────
def cache_prices_yf(ticker: str, start: str, end: str, raw_dir: Path) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance is required: pip install yfinance")
    _ensure_dir(raw_dir)
    pq_path = raw_dir / f"{ticker}.parquet"
    csv_path = raw_dir / f"{ticker}.csv"
    card_path = raw_dir / f"{ticker}.card.json"

    if pq_path.exists() or csv_path.exists():
        try:
            if pq_path.exists():
                df = pd.read_parquet(pq_path)
            else:
                df = pd.read_csv(csv_path, parse_dates=["Date"]).set_index("Date")
            return df
        except Exception:
            pass

    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(-1)
    df = df.reset_index().rename(columns={"Date":"Date"}).set_index("Date").sort_index()
    df.index = pd.to_datetime(df.index)

    engine = _to_parquet_or_csv(df, pq_path)
    card = {
        "ticker": ticker, "start": start, "end": end,
        "vendor": "yfinance", "auto_adjust": True,
        "retrieval_utc": datetime.utcnow().isoformat(),
        "hash_csv": _md5_df(df), "engine": engine
    }
    card_path.write_text(json.dumps(card, indent=2))
    return df

# ──────────────────────────────────────────────────────────────────────────────
# TA and CEEMDAN
# ──────────────────────────────────────────────────────────────────────────────
def ta_features(df: pd.DataFrame) -> pd.DataFrame:
    px = df.copy()
    out = pd.DataFrame(index=px.index)

    out["ret_1"]  = px["Close"].pct_change(1)
    out["ret_5"]  = px["Close"].pct_change(5)
    out["ret_10"] = px["Close"].pct_change(10)

    out["vol_10"] = out["ret_1"].rolling(10).std()
    out["vol_20"] = out["ret_1"].rolling(20).std()

    for w in (10, 20, 50):
        out[f"sma_{w}"] = px["Close"].rolling(w).mean() / px["Close"] - 1.0
        out[f"ema_{w}"] = px["Close"].ewm(span=w, adjust=False).mean() / px["Close"] - 1.0

    # RSI 14
    delta = px["Close"].diff()
    up = delta.clip(lower=0.0).rolling(14).mean()
    down = (-delta.clip(upper=0.0)).rolling(14).mean()
    rs = up / (down + 1e-12)
    out["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = px["Close"].ewm(span=12, adjust=False).mean()
    ema26 = px["Close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    out["macd"] = macd
    out["macd_signal"] = signal
    out["macd_hist"] = macd - signal

    # Bollinger
    ma20 = px["Close"].rolling(20).mean()
    sd20 = px["Close"].rolling(20).std()
    out["bb_upper"] = (ma20 + 2 * sd20) / px["Close"] - 1.0
    out["bb_lower"] = (ma20 - 2 * sd20) / px["Close"] - 1.0
    out["bb_width"] = (2 * sd20) / (ma20 + 1e-12)

    tr = pd.concat([
        (px["High"] - px["Low"]).abs(),
        (px["High"] - px["Close"].shift(1)).abs(),
        (px["Low"]  - px["Close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    out["atr_14"] = tr.rolling(14).mean()

    return out

def maybe_ceemdan_features(series: pd.Series, max_imf=6, trials=100, noise_width=0.05,
                           energy_min=0.02, corr_min=0.10) -> pd.DataFrame:
    if CEEMDAN is None:
        raise ImportError(
            "Missing dependency for CEEMDAN: EMD-signal (PyEMD). "
            "Install with: pip install EMD-signal"
        )

    x = series.values.astype(float)
    ceemdan = CEEMDAN(trials=trials, noise_width=noise_width)
    imfs = ceemdan.ceemdan(x)[:max_imf]

    cols = {}
    for i in range(imfs.shape[0]):
        v = imfs[i, :]
        energy = float(np.mean(v**2))
        if energy < energy_min:
            continue
        mask = ~np.isnan(v)
        corr = float(np.corrcoef(v[mask], x[mask])[0, 1]) if mask.any() else 0.0
        if abs(corr) < corr_min:
            continue
        cols[f"imf_{i+1}"] = v
    return pd.DataFrame(cols, index=series.index)

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    cfg = _get_cfg()
    print(f"PROFILE={cfg['profile']} | ENV-overrides active")

    set_all_seeds(cfg["seed"])

    run_dir = ROOT / "runs" / cfg["profile"]
    out_dir = run_dir / "preprocessed"; _ensure_dir(out_dir)
    raw_dir = ROOT / "data" / "raw";     _ensure_dir(raw_dir)

    macro_csv = ROOT / "macro" / "macro_monthly.csv"
    if cfg["include_macro"]:
        ensure_macro_monthly(macro_csv, cfg["start"], cfg["end"])

    for t in cfg["tickers"]:
        df = cache_prices_yf(t, cfg["start"], cfg["end"], raw_dir)
        needed = [c for c in ["Open","High","Low","Close","Adj Close","Volume"] if c in df.columns]
        df = df[needed].copy()
        if "Adj Close" not in df.columns:
            df["Adj Close"] = df["Close"]

        ta = ta_features(df)
        imf = (maybe_ceemdan_features(df["Close"], **cfg["ceemdan"])
               if cfg["use_ceemdan"] else pd.DataFrame(index=df.index))
        if cfg["include_macro"]:
            macro_daily = macro_daily_from_monthly(macro_csv, df.index)
        else:
            macro_daily = pd.DataFrame(index=df.index)

        feats = pd.concat([ta, imf, macro_daily], axis=1)

        all_nan_cols = feats.columns[feats.isna().all()].tolist()
        if all_nan_cols:
            print(f"[warn] Dropping all-NaN feature columns for {t}: {all_nan_cols}")
            feats = feats.drop(columns=all_nan_cols)

        feats = (
            feats
            .replace([np.inf, -np.inf], np.nan)
            .ffill()
            .bfill()
            .fillna(0.0)
        )

        arr = feats.to_numpy(dtype=float, copy=False)
        if not np.isfinite(arr).all():
            print(f"[warn] Non-finite values remained after cleaning for {t}; applying nan_to_num safeguard")
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            feats = pd.DataFrame(arr, index=feats.index, columns=feats.columns)

        warmup = 60
        feats = feats.iloc[warmup:].copy()
        px   = df.loc[feats.index, "Close"]

        X_raw = feats.values.astype(np.float32)
        y_raw = px.values.astype(np.float32)

        tdir = out_dir / t; _ensure_dir(tdir)
        np.save(tdir / "X_raw.npy", X_raw)
        np.save(tdir / "y_raw.npy", y_raw)

        feats.to_csv(tdir / "ta.csv")
        (tdir / "features_columns.json").write_text(json.dumps(list(feats.columns), indent=2))

        meta = {
            "ticker": t, "profile": cfg["profile"],
            "n_rows": int(X_raw.shape[0]),
            "n_features": int(X_raw.shape[1]) if X_raw.size else 0,
            "include_macro": bool(cfg["include_macro"]),
            "use_ceemdan": bool(cfg["use_ceemdan"]),
            "scaler_type": cfg["scaler_type"],
            "seq_len": cfg["seq_len"],
            "train_end_y": cfg["train_end_y"],
            "seed": cfg["seed"]
        }
        (tdir / "features_meta.json").write_text(json.dumps(meta, indent=2))

        (tdir / "dropped_features.json").write_text(
            json.dumps({"dropped_all_nan": all_nan_cols}, indent=2)
        )

        print(f"[preprocess] {t}: rows={meta['n_rows']}, n_features={meta['n_features']} (macro={meta['include_macro']}, ceemdan={meta['use_ceemdan']})")

    print(f"Preprocessing done -> {out_dir}")

if __name__ == "__main__":
    pd.options.mode.copy_on_write = True
    main()
