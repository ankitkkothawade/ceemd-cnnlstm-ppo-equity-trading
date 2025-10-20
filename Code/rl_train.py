# rl_train.py

import os, json, numpy as np, pandas as pd, yfinance as yf
from typing import Dict, List, Tuple, Callable
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from seed_utils import set_all_seeds

import matplotlib.pyplot as plt
from datetime import timedelta

# ------------- CONFIG -------------
PROFILE         = os.environ.get("PROFILE", "ceemd_cnnlstm_rl_ta_macro")
TICKERS_ENV     = os.environ.get("TICKERS", "AAPL,JPM,AMZN,TSLA,MSFT").split(",")
INITIAL_CASH    = float(os.environ.get("INITIAL_CASH", "10000"))
TRANSACTION_COST= float(os.environ.get("TRANSACTION_COST", "0.001"))
SLIPPAGE_BPS    = float(os.environ.get("SLIPPAGE_BPS", "5")) / 1e4
MIN_CASH_BUFFER = float(os.environ.get("MIN_CASH_BUFFER", "0.02"))
REBAL_THRESH    = float(os.environ.get("REBAL_THRESH", "0.005"))
MAX_WEIGHT      = float(os.environ.get("MAX_WEIGHT", "0.40"))
PPO_STEPS       = int(os.environ.get("PPO_STEPS", "200000"))
SEED            = int(os.environ.get("SEED", "42"))

RUN_DIR         = os.path.join("runs", PROFILE)
IN_DIR          = os.path.join(RUN_DIR, "forecast")
OUT_DIR         = os.path.join(RUN_DIR, "rl")
FIG_DIR         = os.path.join(OUT_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
set_all_seeds(SEED)

PROFILES_FOR_FIGS = {
    "CEEMD–CNN–LSTM + Macro": "ceemd_cnnlstm_rl_ta_macro",
    "CEEMD–CNN–LSTM":         "ceemd_cnnlstm_rl_ta",
    "CNN–LSTM":               "cnnlstm_rl_ta",
    "LSTM":                   "lstm_rl_ta",
}


def _load_states_split() -> Tuple[
    Tuple[pd.DatetimeIndex, Dict[str, np.ndarray]],
    Tuple[pd.DatetimeIndex, Dict[str, np.ndarray]],
    List[str], int]:
    """
    Load per-ticker RL states separately for TRAIN and TEST and align them by the common dates.
    Returns:
      (dates_train, states_train_dict), (dates_test, states_test_dict), tickers, state_dim
    """
    def load_split(suffix: str):
        per_t, valid = {}, []
        for t in TICKERS_ENV:
            tp = os.path.join(IN_DIR, t)
            sp = os.path.join(tp, f"state_{suffix}.npy")
            dp = os.path.join(tp, f"dates_{suffix}.npy")
            if not (os.path.exists(sp) and os.path.exists(dp)):
                continue
            s = np.load(sp)
            d = pd.to_datetime(np.load(dp))
            per_t[t] = {"state": s, "dates": pd.DatetimeIndex(d)}
            valid.append(t)
        if not valid:
            return pd.DatetimeIndex([]), {}, [], []
        dates = None
        for t in valid:
            di = per_t[t]["dates"]
            dates = di if dates is None else dates.intersection(di)
        dates = dates.sort_values()
        aligned, dims = {}, []
        for t in valid:
            di, st = per_t[t]["dates"], per_t[t]["state"]
            loc = pd.Series(np.arange(len(di)), index=di).reindex(dates).values
            aligned[t] = st[loc]
            dims.append(aligned[t].shape[1])
        return dates, aligned, dims, sorted(valid)

    d_tr, st_tr, dims_tr, valid_tr = load_split("train")
    d_te, st_te, dims_te, valid_te = load_split("test")
    if valid_tr and valid_te:
        valid = sorted(list(set(valid_tr).intersection(valid_te)))
    else:
        valid = sorted(list(set(valid_tr + valid_te)))

    max_dim = 0
    for t in set(valid_tr + valid_te):
        if t in st_tr: max_dim = max(max_dim, st_tr[t].shape[1])
        if t in st_te: max_dim = max(max_dim, st_te[t].shape[1])

    def pad(st_dict):
        out = {}
        for t, st in st_dict.items():
            if st.shape[1] < max_dim:
                padv = np.zeros((st.shape[0], max_dim - st.shape[1]), dtype=st.dtype)
                out[t] = np.hstack([st, padv])
            else:
                out[t] = st
        return out

    return (d_tr, pad(st_tr)), (d_te, pad(st_te)), valid, max_dim

def _get_prices(dates: pd.DatetimeIndex, tickers: List[str]) -> pd.DataFrame:
    """Download prices for the given date span; align to dates; forward-fill."""
    if dates.size == 0 or not tickers:
        return pd.DataFrame(index=dates, columns=tickers, dtype="float64")
    start = (dates.min() - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    end   = (dates.max() + pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    frames = []
    for t in tickers:
        df = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
        if df is None or len(df) == 0: 
            s = pd.Series(index=dates, dtype="float64", name=t)
        else:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(-1)
            s = df["Close"].astype("float64").rename(t)
        frames.append(s)
    px = pd.concat(frames, axis=1).sort_index()
    px = px.reindex(dates).ffill()
    return px

# -------------------
# Env
# -------------------
class PortfolioEnv(gym.Env):
    """
    Fill convention: execute at next day's open proxied by today's close with slippage.
    obs = concat(states per ticker) + cash_ratio + current weights
    act = target weights (bounded per-asset); soft-normalized to respect cash buffer
    reward = change in net worth (daily)
    """
    metadata = {"render_modes": []}
    def __init__(self, dates, state_by_t, prices, tickers, state_dim,
                 tc=0.001, slippage=0.0005, cash_buf=0.02, cash0=10000.0,
                 max_weight=0.4, rebalance_thresh=0.005):
        super().__init__()
        self.tickers = sorted(tickers)
        self.dates = dates
        self.state_by_t = state_by_t
        self.prices_df = prices[self.tickers].copy()
        self.tc = float(tc)
        self.slip = float(slippage)
        self.cash_buf = float(cash_buf)
        self.max_w = float(max_weight)
        self.reb_th = float(rebalance_thresh)
        self.n = len(self.tickers)
        self.state_dim = state_dim
        self.step_idx = 0
        self.cash0 = cash0
        self.cash = cash0
        self.shares = np.zeros(self.n, dtype=np.float64)
        self.net = cash0
        obs_len = self.n * self.state_dim + 1 + self.n
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_len,), dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.n,), dtype=np.float32)
        self.history = {"nav": [], "ret": [], "w": [], "dates": []}

    def _obs(self):
        idx = min(self.step_idx, len(self.dates) - 1)
        st = np.hstack([self.state_by_t[t][idx] for t in self.tickers])
        px = self.prices_df.iloc[idx].values
        port_val = self.cash + np.sum(self.shares * px)
        curr_w = (self.shares * px) / max(port_val, 1e-12)
        cash_ratio = self.cash / max(port_val, 1e-12)
        return np.concatenate([st, [cash_ratio], curr_w]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_idx = 0
        self.cash = float(self.cash0)
        self.shares[:] = 0.0
        self.net = float(self.cash0)
        self.history = {"nav": [self.net], "ret": [0.0], "w": [], "dates": [self.dates[0] if len(self.dates) else None]}
        return self._obs(), {}

    def step(self, action):
        w_tgt = np.clip(action, 0.0, 1.0)
        if w_tgt.sum() > 1.0:
            w_tgt = w_tgt / (w_tgt.sum() + 1e-12)
        w_tgt = np.minimum(w_tgt, self.max_w)
        w_tgt = w_tgt * (1.0 - self.cash_buf)  

        idx = min(self.step_idx, len(self.dates) - 1)
        px = self.prices_df.iloc[idx].values
        port_val = self.cash + np.sum(self.shares * px)
        w_cur = (self.shares * px) / max(port_val, 1e-12)

        delta_w = w_tgt - w_cur
        if np.abs(delta_w).sum() > self.reb_th:
            target_dollar = w_tgt * port_val
            target_shares = target_dollar / (px * (1.0 + self.slip))
            trade_shares = target_shares - self.shares
            trade_val = np.sum(np.abs(trade_shares) * px)
            costs = trade_val * (self.tc + self.slip)
            cash_after = self.cash - np.sum(trade_shares * px * (1.0 + self.slip)) - costs
            if cash_after >= 0.0:  
                self.shares += trade_shares
                self.cash = cash_after

        self.step_idx += 1
        done = self.step_idx >= (len(self.dates) - 1) 
        next_idx = min(self.step_idx, len(self.dates) - 1)
        px_next = self.prices_df.iloc[next_idx].values
        nav = self.cash + np.sum(self.shares * px_next)
        ret = (nav / self.history["nav"][-1] - 1.0) if self.history["nav"] else 0.0
        self.net = nav
        self.history["nav"].append(nav)
        self.history["ret"].append(ret)
        self.history["w"].append((self.shares * px_next) / max(nav, 1e-12))
        self.history["dates"].append(self.dates[next_idx])
        reward = ret
        info = {}
        return (self._obs(), reward, done, False, info)

# -------------------
# Metrics
# -------------------
def sharpe_daily(returns: np.ndarray, rf_annual=0.0) -> float:
    if returns.size == 0 or returns.std() == 0:
        return 0.0
    rf_daily = rf_annual / 252.0
    excess = returns - rf_daily
    return float(np.sqrt(252) * excess.mean() / (excess.std() + 1e-12))

def max_drawdown(nav: np.ndarray) -> float:
    if nav.size == 0:
        return 0.0
    cummax = np.maximum.accumulate(nav)
    dd = (cummax - nav) / np.where(cummax == 0, 1, cummax)
    return float(dd.max())

def calmar(nav: np.ndarray, returns: np.ndarray) -> float:
    mdd = max_drawdown(nav)
    ann_ret = (1 + returns).prod() ** (252 / max(len(returns), 1)) - 1
    return float(ann_ret / (mdd + 1e-12)) if mdd > 0 else float("inf")

def hit_ratio(returns: np.ndarray) -> float:
    return float((returns > 0).mean()) if returns.size else 0.0

def turnover(weights: List[np.ndarray]) -> float:
    if len(weights) < 2:
        return 0.0
    diffs = [np.sum(np.abs(weights[i] - weights[i - 1])) for i in range(1, len(weights))]
    return float(np.mean(diffs))

def exposure_mean(weights: List[np.ndarray]) -> float:
    if not weights:
        return 0.0
    return float(np.mean([1.0 - (np.isclose(wt, 0.0).all()) for wt in weights]))

def metric_pack(nav: np.ndarray, rets: np.ndarray, wts: List[np.ndarray], initial_cash: float) -> Dict:
    return {
        "frequency": "daily",
        "risk_free_annual": 0.0,
        "final_nav": float(nav[-1]) if nav.size else initial_cash,
        "total_return": float(nav[-1] / initial_cash - 1.0) if nav.size else 0.0,
        "sharpe": sharpe_daily(rets),
        "max_drawdown": max_drawdown(nav),
        "calmar": calmar(nav, rets),
        "hit_ratio": hit_ratio(rets),
        "exposure_mean": exposure_mean(wts),
        "turnover_mean": turnover(wts),
        "steps": int(len(nav) - 1),
    }

# -------------------
# Simple baselines on TEST
# -------------------
def backtest_weights(prices: pd.DataFrame,
                     weight_rule: Callable[[int, pd.DataFrame], np.ndarray],
                     tc: float, slip: float, cash0: float) -> Dict[str, np.ndarray]:
    """Generic backtester with same fill convention/costs."""
    dates = prices.index
    n = prices.shape[1]
    shares = np.zeros(n, dtype=float)
    cash = cash0
    nav_hist = [cash0]
    ret_hist = [0.0]
    w_hist = []
    for i in range(len(dates) - 1):
        px = prices.iloc[i].values
        port_val = cash + np.sum(shares * px)
        w_tgt = weight_rule(i, prices.iloc[:i+1])
        w_tgt = np.clip(w_tgt, 0.0, 1.0)
        if w_tgt.sum() > 1.0:
            w_tgt = w_tgt / (w_tgt.sum() + 1e-12)

        target_dollar = w_tgt * port_val
        target_shares = target_dollar / (px * (1.0 + slip))
        trade_shares = target_shares - self.shares if "self" in locals() else (target_shares - shares)
        trade_val = np.sum(np.abs(trade_shares) * px)
        costs = trade_val * (tc + slip)
        cash -= np.sum(trade_shares * px * (1.0 + slip)) + costs
        shares += trade_shares

        px_next = prices.iloc[i + 1].values
        nav = cash + np.sum(shares * px_next)
        ret = nav / nav_hist[-1] - 1.0
        nav_hist.append(nav)
        ret_hist.append(ret)
        w_hist.append((shares * px_next) / max(nav, 1e-12))
    return {"nav": np.array(nav_hist), "ret": np.array(ret_hist[1:]), "w": w_hist}

def baseline_buy_hold(prices: pd.DataFrame) -> Dict[str, np.ndarray]:
    n = prices.shape[1]
    px0 = prices.iloc[0].values
    cash0 = INITIAL_CASH
    shares = (cash0 / n) / px0
    cash = 0.0
    nav = [cash0]
    rets = [0.0]
    w_hist = []
    for i in range(1, len(prices)):
        px = prices.iloc[i].values
        val = cash + np.sum(shares * px)
        nav.append(val)
        rets.append(val / nav[-2] - 1.0)
        w_hist.append((shares * px) / max(val, 1e-12))
    return {"nav": np.array(nav), "ret": np.array(rets[1:]), "w": w_hist}

def baseline_equal_weight(prices: pd.DataFrame) -> Dict[str, np.ndarray]:
    n = prices.shape[1]
    def rule(i, px_hist):
        return np.full(n, 1.0 / n)
    return backtest_weights(prices, rule, tc=TRANSACTION_COST, slip=SLIPPAGE_BPS, cash0=INITIAL_CASH)

def baseline_signal_only(prices: pd.DataFrame) -> Dict[str, np.ndarray]:
    n = prices.shape[1]
    rets = prices.pct_change().fillna(0.0)
    def rule(i, px_hist):
        sig = (rets.iloc[i] > 0).astype(float).values
        if sig.sum() == 0:
            return np.zeros(n)
        return sig / sig.sum()
    return backtest_weights(prices, rule, tc=TRANSACTION_COST, slip=SLIPPAGE_BPS, cash0=INITIAL_CASH)

# -------------------
# Main
# -------------------
if __name__ == "__main__":
    (dates_tr, states_tr), (dates_te, states_te), tickers, state_dim = _load_states_split()
    prices_tr = _get_prices(dates_tr, tickers)
    prices_te = _get_prices(dates_te, tickers)

    with open(os.path.join(OUT_DIR, "rl_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "PROFILE": PROFILE, "TICKERS": tickers, "INITIAL_CASH": INITIAL_CASH,
            "TRANSACTION_COST": TRANSACTION_COST, "SLIPPAGE_BPS": SLIPPAGE_BPS*1e4,
            "MIN_CASH_BUFFER": MIN_CASH_BUFFER, "REBAL_THRESH": REBAL_THRESH,
            "MAX_WEIGHT": MAX_WEIGHT, "PPO_STEPS": PPO_STEPS, "SEED": SEED,
            "state_dim": state_dim
        }, f, indent=2)

    if len(dates_te):
        np.save(os.path.join(OUT_DIR, "dates_test.npy"), dates_te.to_numpy())

    env_train = PortfolioEnv(dates_tr, states_tr, prices_tr, tickers, state_dim,
                             tc=TRANSACTION_COST, slippage=SLIPPAGE_BPS,
                             cash_buf=MIN_CASH_BUFFER, cash0=INITIAL_CASH,
                             max_weight=MAX_WEIGHT, rebalance_thresh=REBAL_THRESH)
    env_test = PortfolioEnv(dates_te, states_te, prices_te, tickers, state_dim,
                            tc=TRANSACTION_COST, slippage=SLIPPAGE_BPS,
                            cash_buf=MIN_CASH_BUFFER, cash0=INITIAL_CASH,
                            max_weight=MAX_WEIGHT, rebalance_thresh=REBAL_THRESH)

    model = PPO("MlpPolicy", env_train, seed=SEED, verbose=0)
    model.learn(total_timesteps=PPO_STEPS)
    model.save(os.path.join(OUT_DIR, "ppo.zip"))

    obs, _ = env_test.reset(seed=SEED)
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env_test.step(action)

    nav = np.array(env_test.history["nav"], dtype=float)
    rets = np.array(env_test.history["ret"][1:], dtype=float)  # drop first 0
    wts  = env_test.history["w"]
    ppo_metrics = metric_pack(nav, rets, wts, INITIAL_CASH)

    bh = baseline_buy_hold(prices_te)
    ew = baseline_equal_weight(prices_te)
    sg = baseline_signal_only(prices_te)

    baselines = {
        "buy_hold":     metric_pack(bh["nav"], bh["ret"], bh["w"], INITIAL_CASH),
        "equal_weight": metric_pack(ew["nav"], ew["ret"], ew["w"], INITIAL_CASH),
        "signal_only":  metric_pack(sg["nav"], sg["ret"], sg["w"], INITIAL_CASH),
    }

    results = {
        "ppo": ppo_metrics,
        "baselines": baselines,
        "tickers": tickers,
        "steps": int(len(nav)-1),
        "notes": {
            "frequency": "daily",
            "rf_assumption": "0%",
            "fill_convention": "next-day open proxied by close with slippage",
        }
    }
    with open(os.path.join(OUT_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    if len(env_test.history["dates"]) > 1:
        hist_df = pd.DataFrame({
            "date": pd.to_datetime(env_test.history["dates"][1:], utc=False),
            "nav": np.array(env_test.history["nav"][1:], dtype=float),
            "ret": np.array(env_test.history["ret"][1:], dtype=float)
        })
        hist_df.to_csv(os.path.join(OUT_DIR, "ppo_history_test.csv"), index=False)

        try:
            weights_arr = np.vstack(env_test.history["w"]) if env_test.history["w"] else np.empty((0, len(tickers)))
            if weights_arr.size > 0:
                weights_df = pd.DataFrame(weights_arr, index=pd.to_datetime(env_test.history["dates"][1:], utc=False), columns=tickers)
                weights_df.index.name = "date"
                weights_df.to_csv(os.path.join(OUT_DIR, "ppo_weights_test.csv"))
        except Exception as e:
            with open(os.path.join(OUT_DIR, "weights_export_error.txt"), "w") as fe:
                fe.write(str(e))

    def _save_hist(name, hist):
        if prices_te is None or prices_te.empty:
            return
        dates_idx = prices_te.index
        if len(dates_idx) > 1:
            _df = pd.DataFrame({
                "date": pd.to_datetime(dates_idx[1:], utc=False),
                "nav": hist["nav"][1:],     
                "ret": hist["ret"]
            })
            _df.to_csv(os.path.join(OUT_DIR, f"{name}_history_test.csv"), index=False)

    _save_hist("buy_hold", bh)
    _save_hist("equal_weight", ew)
    _save_hist("signal_only", sg)

    flat = dict(ppo_metrics)
    flat.update({"profile": PROFILE, "algo": "ppo"})
    pd.DataFrame([flat]).to_csv(os.path.join(OUT_DIR, "rl_metrics.csv"), index=False)

    rows = []
    for k, met in baselines.items():
        r = dict(met)
        r.update({"profile": PROFILE, "algo": k})
        rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "baseline_metrics.csv"), index=False)

    print("✅ RL done →", results["ppo"])

   