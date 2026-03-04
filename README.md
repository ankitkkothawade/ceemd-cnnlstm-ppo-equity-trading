# 📈 CEEMD–CNN–LSTM–PPO: Frequency-Aware Equity Forecasting & Portfolio Management  
**Hybrid Deep Learning Pipeline for Financial Time-Series Forecasting**  
*CEEMDAN + CNN–LSTM + PPO Portfolio Optimisation*

---

## 🚀 Project Highlights

- Built a hybrid ML pipeline combining **CEEMDAN signal decomposition, CNN–LSTM forecasting, and PPO reinforcement learning**
- Designed a **walk-forward evaluation framework** to prevent look-ahead bias in financial time-series forecasting
- Integrated **macro + technical features** across 5 equities using 24 years of historical data
- Converted predictive signals into **portfolio allocation decisions** using reinforcement learning under realistic trading frictions
- Demonstrated improved **forecast stability and risk-adjusted portfolio performance**

---

## 🏗 System Architecture

![System Architecture](docs/architecture.png)

**Flow:**

Market Data  
↓  
CEEMDAN Decomposition  
↓  
Feature Engineering  
↓  
CNN–LSTM Forecast Model  
↓  
Predicted Returns  
↓  
PPO Agent  
↓  
Portfolio Allocation  

---

# ⚡ TL;DR

## Problem
Daily equity prices are **noisy, non-stationary and regime-dependent**.  
Traditional models struggle to convert short-term predictive edges into robust portfolio returns.

## Solution
Decompose prices into intrinsic mode functions with **CEEMDAN** to isolate meaningful signals, forecast with a **CNN–LSTM hybrid**, and optimise allocations with a **PPO trading agent** under realistic transaction costs.

## Impact
The hybrid system improved:
- Forecast accuracy (**lower RMSE / MAE**)
- Risk-adjusted returns (**higher Sharpe ratio, lower drawdowns**)
- Robustness across **multiple market regimes**

This project was delivered as part of my **MSc dissertation at the University of Strathclyde (Aug 2025)** and emphasises:
- reproducible research  
- business-relevant evaluation  
- production-ready transparency  

---

## 📈 Example Output (Quick Demo)

Forecast vs actual (example):

![Prediction vs Actual](docs/prediction_plot.png)

Portfolio equity curve (example):

![Equity Curve](docs/equity_curve.png)

---

# 🧠 Why This Matters

Financial time series are notoriously **noisy and non-stationary**.  
A model that works during one regime often fails in another.

Furthermore, **predictive accuracy alone does not translate into trading profits** when transaction costs and risk constraints are considered.

This repository tackles these challenges by integrating signal processing, deep learning, and reinforcement learning.

---

## Denoising
Using **Complete Ensemble Empirical Mode Decomposition with Adaptive Noise (CEEMDAN)** to extract multi-scale components that retain meaningful price dynamics while filtering out noise.

## Forecasting
Building a **CNN–LSTM architecture** that captures:
- local patterns via **convolutions**
- long-term dependencies via **recurrent memory**

## Decision Making
Training a **PPO reinforcement learning agent** to dynamically adjust portfolio weights while accounting for:
- transaction costs  
- slippage  
- portfolio constraints  

## Validation
Ensuring reliability with:
- walk-forward evaluation  
- ablation studies  
- paired bootstrap confidence intervals  
- leakage-safe pipelines  

---

# 🔬 Key Findings

### Decomposition improves stability
Models trained on **CEEMDAN-derived components** show smoother error curves and lower **RMSE / MAE**.

### Macro and technical features matter
Adding macro indicators (**VIX, CPI, unemployment**) and technical indicators improves directional accuracy and risk-adjusted performance.

### Reinforcement learning closes the loop
**PPO converts predictive signals into profitable trading decisions** under realistic trading costs.

### Reproducibility matters
Strict seed control and leakage safeguards ensure **repeatable experimental results**.

---

# 📂 Project Structure

```text
ceemd-cnnlstm-ppo/
│
├── preprocess.py        # Data acquisition, CEEMDAN decomposition, feature generation
├── forecast.py          # CNN–LSTM model training and forecasting
├── rl_train.py          # PPO trading agent and environment
├── ablation_runner.py   # Orchestrates experimental profiles
├── seed_utils.py        # Deterministic seeding utilities
│
├── requirements.txt     # Python dependencies
├── How to Run.txt       # Minimal execution instructions
├── Report.docx          # Full dissertation report
└── README.md            # Project documentation
