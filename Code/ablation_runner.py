# ablation_runner.py
import os, sys, json, yaml, subprocess, pandas as pd

def run_cmd(env, pyfile):
    r = subprocess.run(
        [sys.executable, pyfile],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    print(r.stdout)
    if r.returncode != 0:
        raise RuntimeError(f"{pyfile} failed")

def _load_forecast_metrics(run_dir: str) -> dict:
    fjson = os.path.join(run_dir, "forecast", "pooled_metrics.json")
    if not os.path.exists(fjson):
        return {}
    with open(fjson, "r") as f:
        m = json.load(f)
    # normalize keys
    return {
        "rmse": m.get("rmse"),
        "mae": m.get("mae"),
        "da": m.get("da"),
    }

def _load_rl_metrics(run_dir: str) -> dict:
    rl_dir = os.path.join(run_dir, "rl")
    csv_path = os.path.join(rl_dir, "rl_metrics.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if not df.empty:
            row = df.iloc[0].to_dict()
            return {
                "return": row.get("total_return"),
                "sharpe": row.get("sharpe"),
                "max_dd": row.get("max_drawdown"),
                "final_nav": row.get("final_nav"),
            }
    rjson = os.path.join(rl_dir, "results.json")
    if os.path.exists(rjson):
        with open(rjson, "r") as f:
            j = json.load(f)
        ppo = j.get("ppo", {})
        return {
            "return": ppo.get("total_return"),
            "sharpe": ppo.get("sharpe"),
            "max_dd": ppo.get("max_drawdown"),
            "final_nav": ppo.get("final_nav"),
        }
    # Nothing found
    return {"return": None, "sharpe": None, "max_dd": None, "final_nav": None}

if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/exp.yaml"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    g = cfg["global"]
    base_env = os.environ.copy()
    base_env.update({
        "TICKERS": ",".join(g["tickers"]),
        "START": g["start"], "END": g["end"],
        "TRAIN_END_Y": str(g["train_end_y"]),
        "SEQ_LEN": str(g["seq_len"]),
        "SCALER_TYPE": g["scaler_type"],
        "SEED": str(g["seed"]),
        "PPO_STEPS": str(g["ppo_steps"]),
        "TRANSACTION_COST": str(g["transaction_cost"]),
        "MIN_CASH_BUFFER": str(g["min_cash_buffer"]),
        "INITIAL_CASH": str(g["initial_cash"]),
        "MAX_IMF": str(g["ceemdan"]["max_imf"]),
        "TRIALS": str(g["ceemdan"]["trials"]),
        "NOISE_WIDTH": str(g["ceemdan"]["noise_width"]),
        "ENERGY_MIN": str(g["ceemdan"]["energy_min"]),
        "CORR_MIN": str(g["ceemdan"]["corr_min"]),
        "INCLUDE_RSI": "1", "INCLUDE_MACD": "1", "INCLUDE_BB": "1", "INCLUDE_VOL": "1",
        "INCLUDE_PRED_CHANNEL": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
    })

    rows = []
    for profile, p in cfg["profiles"].items():
        env = base_env.copy()
        env.update({
            "PROFILE": profile,
            "USE_CEEMDAN": "1" if p["use_ceemdan"] else "0",
            "INCLUDE_MACRO": "1" if p["include_macro"] else "0",
            "MODEL_TYPE": p["model_type"],
        })

        print(f"\n=== Running profile: {profile} ===")
        run_cmd(env, "preprocess.py")
        run_cmd(env, "forecast.py")
        run_cmd(env, "rl_train.py")

        run_dir = os.path.join("runs", profile)
        fmetrics = _load_forecast_metrics(run_dir)
        rmetrics = _load_rl_metrics(run_dir)

        row = {"profile": profile}
        row.update(fmetrics)
        row.update(rmetrics)
        rows.append(row)

    df = pd.DataFrame(rows)
    for c in ["rmse", "mae", "da", "return", "sharpe", "max_dd", "final_nav"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    os.makedirs("runs", exist_ok=True)
    df.set_index("profile").to_csv("runs/ablation_summary.csv")
    print("\nAblation summary (runs/ablation_summary.csv):")
    print(df.set_index("profile").to_markdown(floatfmt=".4f"))
