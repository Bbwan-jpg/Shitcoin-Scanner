# Shitcoin-Scanner — Solana Pump & Dump Detection with ML

A real-time crypto market surveillance system designed to detect **pump-and-dump patterns** on the Solana ecosystem using:

- Market microstructure metrics (price, liquidity, buy pressure, volume ratios)
- A supervised Machine Learning risk model (Logistic Regression, scikit-learn)
- A full interactive Streamlit dashboard
- A local SQL database for users, watchlists, presets, and scan history

Data sources:
- DexScreener API (pairs, metrics, socials)
- Birdeye API (OHLCV candles)
- Optional Binance OHLCV (training dataset generation via CCXT)

---

## What this project does

This system continuously scans newly created Solana pairs and:

1. Extracts statistical metrics from DexScreener
2. Computes a heuristic score based on market behavior
3. Applies a trained ML model to estimate **pump/dump risk probability**
4. Displays candidates in a Streamlit dashboard
5. Allows live tracking, watchlisting, and history logging
6. Provides model explainability (feature contributions per token)

---

## Project Structure

| File | Role |
|-----|------|
| `app_streamlit.py` | Main Streamlit dashboard (UI + live monitoring) |
| `sol_watcher_stat.py` | Statistical engine & scoring logic |
| `risk_model.py` | ML inference utilities |
| `pump_learn.py` | ML training pipeline from labeled events |
| `db.py` | SQLite DB: users, presets, watchlist, history |
| `Solana.py` | DexScreener data extraction & metrics |
| `model_pump.pkl` | Trained ML model |
| `tests/` | Pytest test suite |

---

## Requirements

- Python **3.10+**
- pip
- Optional: Birdeye API key for Plotly candles

---

## 📦 Installation

```bash
git clone https://github.com/Bbwan-jpg/Shitcoin-Scanner.git
cd Shitcoin-Scanner

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
