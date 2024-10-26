# 🏦 Crypto Pair Trading Strategy for Statistical Arbitrage

This repository contains a Python implementation of a pair trading strategy aimed at profiting from the mean-reverting relationship between two cryptocurrency pairs, BTC and ETH. By leveraging statistical indicators, this strategy identifies potential entry and exit points based on the relative strength of BTC and ETH prices. The `CryptoDataFetcher` class pulls historical data from Binance, while `PairTradingStrategy` implements the trading logic and visualization.

---

## ⚙️ Features

1. **Data Fetching** - Retrieves and stores historical BTC and ETH price data from Binance.
2. **Statistical Indicators** - Utilizes Z-scores, rolling means, exponential weighted moving average (EWMA), and RSI for signal generation.
3. **Dynamic Threshold Adjustments** - Adapts Z-score thresholds based on market volatility to optimize entry and exit signals.
4. **Position Management** - Simulates P&L and Mark-to-Market (MtM) values for each trading decision.
5. **Detailed Visualization** - Visualizes trading signals, Z-score dynamics, and the resulting PnL for in-depth analysis.

---

## 🛠️ Code Structure

- `CryptoDataFetcher`: A utility to fetch BTC and ETH historical data from Binance and save it locally in HDF5 format for faster access.
- `PairTradingStrategy`: Implements the core statistical arbitrage trading logic, performing the following:
  - Calculates the rolling Z-score of the BTC/ETH price ratio.
  - Generates trade signals based on threshold-crossing events.
  - Manages positions and computes P&L in response to market conditions.

---

## 📈 Strategy Explanation

The pair trading strategy exploits mean-reverting tendencies between BTC and ETH prices. Key components include:

- **Z-Score Calculation**: Tracks the deviation of BTC/ETH ratios from the mean, helping identify overbought/oversold conditions.
- **Dynamic Thresholds**: Adjusts entry and exit levels according to recent price volatility.
- **RSI Filtering**: Uses Relative Strength Index (RSI) to filter trades, avoiding false signals in strongly trending markets.

---

## 🔧 Requirements

- Python 3.7+
- Libraries: `pandas`, `numpy`, `matplotlib`, `requests`, `ta`, `tqdm`, `logging`
- Binance API access

Install dependencies with:
```bash
pip install -r requirements.txt
