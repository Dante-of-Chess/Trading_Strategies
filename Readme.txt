# Crypto Pair Trading Strategy

This repository contains an implementation of a **Crypto Pair Trading Strategy** using BTC and ETH historical data from the Binance API. The strategy leverages statistical analysis, such as Z-scores and RSI, to identify pair trading opportunities.

## Features

- **Data Fetching**: Pulls BTC and ETH price data from Binance at 1-minute intervals.
- **Pair Trading Logic**: Uses mean-reversion principles with Z-score and dynamic thresholds.
- **Visualization**: Plots trading signals, Z-scores, PnL, and RSI for easy analysis.
- **Customization**: Allows fine-tuning of parameters such as Z-score threshold, EMA windows, and RSI levels.

## Project Structure

- `CryptoDataFetcher`: Fetches and updates crypto data.
- `PairTradingStrategy`: Executes the trading strategy and visualizes results.

## Requirements

Install the required libraries with:
```bash
pip install requests pandas matplotlib ta tqdm
