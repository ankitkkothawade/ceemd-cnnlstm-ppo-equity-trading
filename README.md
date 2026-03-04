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


---

# ⚡ TL;DR

## Problem

Daily equity prices are **noisy, non-stationary, and regime-dependent**.  
Traditional models often struggle to convert short-term predictive edges into **robust portfolio returns**.

## Solution

This system integrates:

- **CEEMDAN signal decomposition** to separate noise from signal  
- **CNN–LSTM forecasting models** to capture spatial and temporal patterns  
- **PPO reinforcement learning agents** to convert predictions into portfolio decisions  

All evaluated under **realistic trading costs and portfolio constraints**.

## Impact

The hybrid system demonstrated:

- Lower **forecast error (RMSE / MAE)**
- Higher **Sharpe ratio**
- Lower **maximum drawdowns**
- Robust performance across **multiple market regimes**

This work was completed as part of my **MSc dissertation at the University of Strathclyde (Aug 2025)** and emphasises:

- reproducible research  
- business-relevant evaluation  
- production-ready transparency  

---

# 📈 Example Output

### Forecast vs Actual Price Movement

![Prediction vs Actual](docs/prediction_plot.png)

### Portfolio Equity Curve

![Equity Curve](docs/equity_curve.png)

---

# 🧠 Why This Matters

Financial time series are notoriously **noisy and non-stationary**.

A forecasting model that performs well in one regime often **fails in another**.

Additionally, **forecast accuracy alone does not translate into trading profits** when transaction costs, slippage, and portfolio constraints are considered.

This repository addresses these issues by integrating **signal processing, deep learning, and reinforcement learning into a single decision pipeline**.

---

## Signal Denoising

The system uses **Complete Ensemble Empirical Mode Decomposition with Adaptive Noise (CEEMDAN)** to extract **multi-scale intrinsic mode functions (IMFs)** that isolate meaningful market dynamics while filtering out noise.

---

## Forecasting

A **CNN–LSTM hybrid architecture** captures:

- local temporal patterns via **convolutional layers**
- long-term dependencies via **LSTM memory**

---

## Decision Making

A **Proximal Policy Optimization (PPO)** reinforcement learning agent dynamically adjusts portfolio weights while accounting for:

- transaction costs  
- slippage  
- portfolio constraints  

---

## Validation

Experimental reliability is ensured through:

- walk-forward evaluation  
- ablation studies  
- paired bootstrap confidence intervals  
- leakage-safe pipelines  

---

# 🔬 Key Findings

### Decomposition improves stability

Models trained on **CEEMDAN-derived components** show smoother error curves and lower **RMSE / MAE**.

### Macro and technical features matter

Adding macro indicators (**VIX, CPI, unemployment**) and technical indicators improves directional accuracy and **risk-adjusted portfolio performance**.

### Reinforcement learning closes the loop

The **PPO trading agent converts predictive signals into profitable allocation decisions** under realistic trading costs.

### Reproducibility matters

Strict seed control and leakage safeguards ensure **repeatable results across experimental runs**.

---

# 📂 Project Structure

```
ceemd-cnnlstm-ppo/
│
├── preprocess.py
│   Data acquisition, CEEMDAN decomposition, feature engineering
│
├── forecast.py
│   CNN–LSTM training and forecasting
│
├── rl_train.py
│   PPO trading agent and custom Gym environment
│
├── ablation_runner.py
│   Orchestrates experimental profiles
│
├── seed_utils.py
│   Deterministic seeding utilities
│
├── requirements.txt
│   Python dependencies
│
├── How to Run.txt
│   Minimal execution instructions
│
├── Report.docx
│   Full dissertation report
│
└── README.md
│   Project documentation
```

---

# 🧪 Experimental Setup

## Assets

- AAPL  
- AMZN  
- TSLA  
- JPM  
- MSFT  

**Time horizon:** 2000 – 2024

---

## Data Sources

- **Yahoo Finance** — equity price data  
- **FRED** — macroeconomic indicators  

---

## Feature Engineering

### CEEMDAN Intrinsic Mode Functions (IMFs)

Filtered using:

- energy threshold  
- correlation threshold  

---

### Technical Indicators

- RSI  
- MACD  
- SMA / EMA  
- Bollinger Bands  
- ATR  
- volatility metrics  

---

### Macroeconomic Features

- VIX  
- CPI YoY  
- unemployment rate  
- output gap proxy  

---

### Leakage Safeguards

- release-lag alignment  
- warm-up trimming  
- train-only scaling  

---

# 🤖 Forecasting Model

## CNN–LSTM Architecture

```
Conv1D → MaxPool → LSTM → Dense
```

### Configuration

- Loss: **MSE**
- Optimizer: **Adam**
- Learning rate: **1e-4**
- Regularisation: **Dropout**

### Training Strategy

- walk-forward validation

---

# 🧠 Reinforcement Learning Agent

### Algorithm

**Proximal Policy Optimization (PPO)**

---

### Action Space

Continuous **portfolio weights** for each asset.

---

### Portfolio Constraints

- Cash buffer: **2%**
- Max allocation per asset: **40%**
- Rebalance threshold: **0.5%**

---

### Trading Costs

- **10 bps transaction cost**
- **5 bps slippage**

---

### Reward

Daily **portfolio NAV change**.

---

### Evaluation Metrics

- Sharpe Ratio  
- Maximum Drawdown  
- Cumulative Return  
- Turnover  

---

# 📊 Evaluation & Results

| Metric | Baseline (LSTM) | Hybrid (CEEMDAN + CNN–LSTM + PPO) | Outcome |
|------|------|------|------|
| RMSE | Higher | Lower | Improved forecasting accuracy |
| Sharpe Ratio | Moderate | Higher | Better risk-adjusted returns |
| Max Drawdown | Larger | Smaller | More stable equity curve |
| Cumulative Return | Baseline | Higher | Outperformed buy-and-hold |

Exact numerical results vary by asset and market regime.  
See the **full dissertation report** for detailed experimental results.

---

# 🚀 Running the Experiments

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Full Experiment Suite

```bash
python ablation_runner.py
```

All artefacts (models, scalers, logs, metrics) are automatically stored for **full reproducibility**.

---

# 🛠 Extending the Framework

You can adapt the system to other markets or assets.

Possible extensions:

- Add additional equities or ETFs in `preprocess.py`
- Modify feature engineering
- Adjust model architecture in `forecast.py`
- Replace PPO with alternative RL algorithms:

**Possible alternatives**

- DQN  
- SAC  
- TD3  

---

# 💡 Lessons Learned

- Financial forecasting models easily **overfit without strict walk-forward validation**.
- Decomposition techniques like **CEEMDAN help stabilise deep learning forecasts** by separating signal from noise.
- Predictive accuracy alone does **not guarantee trading profitability** without incorporating costs and portfolio constraints.
- Reinforcement learning works best when combined with **structured predictive signals rather than raw price data**.

---

# ⚠ Limitations

- Backtesting results depend on historical market regimes and may not fully generalise to future conditions.
- Transaction cost assumptions are simplified relative to real trading environments.
- Reinforcement learning policies require careful reward design to avoid unstable training.
- The system currently runs as a **research pipeline rather than a production API service**.

---

# 👤 About Me

**Ankit Kothawade**  
MSc Advanced Computer Science with Data Science  
University of Strathclyde

I build machine learning systems for **forecasting, decision support, and analytical automation**.

### Interests

- Time-series forecasting  
- Risk modelling  
- Retrieval-augmented ML systems  

Previously worked at **Cognizant**, building SQL-driven data pipelines and analytical workflows.

📧 ankitkkothawade@gmail.com  

🔗 https://linkedin.com/in/ankit-kothawade  

💻 https://github.com/ankitkothawade

---

> “I build ML systems that bridge prediction and decision-making and translate research into practical tools that deliver measurable business value.”

