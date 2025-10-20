
# 📈 CEEMD–CNN–LSTM–PPO: A Hybrid Deep Learning Framework for Equity Forecasting and Trading  

This repository accompanies the dissertation  
**“A CEEMD–CNN–LSTM–PPO Pipeline for Daily Equity Portfolios”**  
submitted in partial fulfilment of the requirements for the *MSc in Advanced Computer Science with Data Science* at the *University of Strathclyde* (August 2025).  

The work investigates the integration of **frequency-aware signal decomposition**, **deep learning sequence modelling**, and **reinforcement learning** for realistic, cost-aware portfolio management.  
All experiments are fully reproducible and designed for deployment-grade transparency.

---

## 🧩 Overview

Financial time series are noisy, non-stationary, and regime-dependent.  
This project develops and evaluates an end-to-end pipeline that integrates:

| Component | Purpose | Methodology |
|------------|----------|-------------|
| **CEEMDAN** | Adaptive denoising and multi-scale signal decomposition | Complete Ensemble Empirical Mode Decomposition with Adaptive Noise |
| **CNN–LSTM Forecaster** | Short-term price prediction | Convolutional feature extraction + gated temporal memory |
| **Reinforcement Learning Agent (PPO)** | Dynamic portfolio allocation under frictions | Proximal Policy Optimization via Stable-Baselines3 |
| **Ablation Framework** | Empirical validation | Controlled toggling of decomposition, macro, and technical features |

---

## 🎯 Research Objectives

1. Evaluate whether **CEEMDAN-based decomposition** enhances deep learning forecast accuracy.  
2. Quantify the contribution of **macroeconomic and technical indicators** to daily predictive performance.  
3. Compare **CNN–LSTM hybrids** against LSTM-only and non-decomposed baselines.  
4. Examine whether **reinforcement learning** improves portfolio returns under realistic costs.  
5. Ensure **reproducibility**, **leakage-safe design**, and **walk-forward evaluation** across multiple market regimes.

---

## 📂 Repository Structure

```

ceemd-cnnlstm-ppo/
│
├── preprocess.py             # Data acquisition, CEEMDAN decomposition, feature generation
├── forecast.py               # CNN–LSTM model training and forecasting
├── rl_train.py               # PPO trading agent and environment
├── ablation_runner.py        # Orchestration of all experimental profiles
├── seed_utils.py             # Deterministic seeding and reproducibility utilities
│
├── requirements.txt          # Python dependencies
├── How to Run.txt            # Minimal execution instructions
│
├── Report.docx               # Full dissertation report
└── README.md                 # Project documentation

````

---

## 🧮 Experimental Design

### Dataset
- **Universe:** AAPL, AMZN, TSLA, JPM, MSFT  
- **Horizon:** 2000 – 2024 (Train 2000–2022  |  Test 2023–2024)  
- **Data Sources:** Yahoo Finance (prices) and FRED (macro indicators)  
- **Frequency:** Daily; aligned “as-of” macro updates to avoid look-ahead bias  

### Features
- CEEMD-derived Intrinsic Mode Functions (filtered by energy ≥ 0.02, |corr| ≥ 0.10)  
- Compact technical indicators (RSI, MACD, SMA/EMA, Bollinger Bands, ATR, volatility metrics)  
- Macroeconomic variables (VIX, CPI YoY, Unemployment, Output Gap proxy)  
- Leakage safeguards: release-lag alignment, warm-up trimming, train-only scaling  

### Forecasting Model
- **Architecture:** Conv1D(32, kernel 3) → MaxPool → LSTM(64) → Dense(1)  
- **Loss:** MSE  |  Optimizer:** Adam (1e-4)  |  Regularization:** Dropout (0.2)  
- **Training protocol:** Early Stopping (patience 8), ReduceLROnPlateau (factor 0.5), walk-forward validation  

### Reinforcement Learning
- **Algorithm:** PPO (MlpPolicy, Stable-Baselines3)  
- **Action space:** Continuous target weights [0, 1] per asset  
- **Risk constraints:** Cash buffer 2 %, Per-asset cap 40 %, Rebalance threshold 0.5 %  
- **Transaction costs:** 10 bps per leg + 5 bps slippage  
- **Reward:** Daily ΔNAV (net of costs)  
- **Metrics:** Sharpe Ratio, Max Drawdown, Cumulative Return, Turnover  

### Ablation Profiles
| Profile | Decomposition | Technical | Macro | RL |
|----------|---------------|-----------|-------|----|
| `ceemd_cnnlstm_rl_ta_macro` | ✅ | ✅ | ✅ | ✅ |
| `ceemd_cnnlstm_rl_ta` | ✅ | ✅ | ✖ | ✅ |
| `cnnlstm_rl_ta` | ✖ | ✅ | ✖ | ✅ |
| `lstm_rl_ta` | ✖ | ✅ | ✖ | ✅ |

---

## 📊 Evaluation Framework

**Forecasting metrics:** RMSE  |  MAE  |  Directional Accuracy (DA)  
**Trading metrics:** Net Asset Value, Sharpe Ratio, Max Drawdown, Calmar Ratio  

Performance is assessed across market regimes (bullish, sideways, volatile) using paired bootstrap confidence intervals for statistical significance.

---

## 📈 Key Findings

- **CEEMD decomposition** improved forecast stability and reduced RMSE/MAE across all assets.  
- **Macro-augmented hybrids** enhanced risk-adjusted performance by reducing drawdowns.  
- **PPO trading agents** converted modest predictive edges into statistically significant portfolio gains.  
- **Walk-forward reproducibility** confirmed the robustness of each experimental variant.  
- **Live paper-trading** validated that back-tested dynamics generalize under real-time execution.  

---

## 🧰 Software Environment

| Category | Tool / Library |
|-----------|----------------|
| **Core** | Python 3.9 + |
| **Data Handling** | NumPy, Pandas, yFinance, pandas-datareader |
| **Modeling** | TensorFlow, PyTorch |
| **Signal Processing** | EMD-signal (CEEMDAN implementation) |
| **Reinforcement Learning** | Gymnasium, Stable-Baselines3 |
| **Utilities** | PyYAML, scikit-learn |
| **Reproducibility** | Fixed seeds (`seed_utils.py`), JSON configs, audit logs |

Install dependencies:
```bash
pip install -r requirements.txt
````

---

## ▶️ Execution

```bash
# Run full experiment suite
python ablation_runner.py
```

All intermediate artifacts (scalers, checkpoints, metrics, logs) are automatically versioned for audit and replication.

---


## 🧭 Research Impact

This work demonstrates that **frequency-aware hybrid models**, when coupled with **cost-aware reinforcement learning**, can close the gap between **forecasting** and **decision-making** in financial markets.
It contributes a reproducible, extensible framework for academic research and applied quantitative finance.

---

## 👤 Author

**Ankit Kothawade**
*MSc Advanced Computer Science with Data Science*
University of Strathclyde

📧 [ankitkkothawade@gmail.com](mailto:ankitkkothawade@gmail.com)
🔗 [linkedin.com/in/ankit-kothawade](https://www.linkedin.com/in/ankit-kothawade)
💻 [github.com/ankitkothawade](https://github.com/ankitkothawade)

> “Bridging signal decomposition, deep learning, and reinforcement learning for transparent, reproducible financial AI.”

---
