# Crypto Pair Trading Strategy

This repository implements a **Statistical Arbitrage** strategy for crypto pairs using BTC and ETH price data from Binance's API. The strategy leverages statistical analysis, such as Z-scores, dynamic thresholds, and RSI, to find and execute profitable trading signals.

---

## 🚀 Features

- **Data Fetching**: Pulls historical BTC and ETH price data from Binance at 1-minute intervals.
- **Statistical Arbitrage Logic**: Employs mean-reversion techniques based on Z-scores and dynamic EWMA thresholds for pair trading.
- **Visualization**: Generates plots for Z-scores, PnL, RSI, and price ratios, making analysis straightforward.
- **Parameter Customization**: Adjust parameters like Z-score thresholds, EMA windows, and RSI levels to fine-tune the strategy.

---

## 📂 Project Structure

- **`CryptoDataFetcher`**: Fetches and updates crypto price data and saves it locally.
- **`PairTradingStrategy`**: Implements the statistical arbitrage logic, with methods for backtesting and visualization.

---

## 📦 Requirements

Install dependencies using:
```bash
pip install requests pandas matplotlib ta tqdm
