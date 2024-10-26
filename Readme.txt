Crypto Pair Trading Strategy
Welcome to the Crypto Pair Trading Strategy project! This codebase implements a pair trading strategy using historical price data for Bitcoin (BTC) and Ethereum (ETH) obtained from the Binance API. The strategy uses advanced statistical techniques, including Z-scores, dynamic thresholds, and RSI indicators, to execute trades and visualize outcomes.

🚀 Features
Data Fetching: Automatically fetches BTC and ETH price data from Binance at a 1-minute interval.
Pair Trading Strategy: Calculates ratios between BTC and ETH prices and performs mean-reversion trading based on Z-score thresholds and dynamic EWMA thresholds.
Visualization: Comprehensive plotting of trading signals, Z-scores, PnL, and RSI.
Easy to Customize: Adjust parameters like Z-score thresholds, EMA windows, and RSI levels for fine-tuning.
📂 Project Structure
CryptoDataFetcher: A class for fetching and saving crypto data locally.
PairTradingStrategy: Implements the trading strategy with backtesting capability and visualization.
📜 Requirements
Python Libraries: requests, pandas, matplotlib, ta, tqdm
Binance API: Access to Binance’s free API for data retrieval.
Install dependencies with:

bash
Copy code
pip install requests pandas matplotlib ta tqdm
📊 Usage Guide
1. Fetch Data
The CryptoDataFetcher class can retrieve historical BTC and ETH prices and save them locally. If a previous file exists, it updates only the latest data.

2. Execute Strategy
The PairTradingStrategy class takes in BTC and ETH data and uses Z-scores and RSI thresholds for pair trading. Customize parameters like Z-score thresholds, EMA windows, and RSI levels to fit market conditions.

3. Visualize Results
The strategy’s results include charts for price ratios, Z-scores, PnL, and RSI, enabling clear analysis of the strategy’s performance.

Example
python
Copy code
# Initialize Data Fetcher
data_fetcher = CryptoDataFetcher()
btc_data = data_fetcher.get_data('BTCUSDT', Pull_latest=True)
eth_data = data_fetcher.get_data('ETHUSDT', Pull_latest=True)

# Preprocess and Merge Data
btc_data['Date'] = pd.to_datetime(btc_data['Date'])
eth_data['Date'] = pd.to_datetime(eth_data['Date'])
btc_data = btc_data.rename(columns={'Close': 'BTC'})
eth_data = eth_data.rename(columns={'Close': 'ETH'})
merged_data = pd.merge(btc_data, eth_data, on='Date', how='left').dropna().set_index('Date')

# Define and Execute Strategy
strategy = PairTradingStrategy(merged_data, z_score_threshold=1, window1=28800, window2=7200, look_back=60*12, rsi_up=50.5, rsi_down=49.5)
strategy.execute_strategy()
strategy.visualize_strategy()
📈 Output
The output includes several charts:

BTC and ETH Pair Ratio: Monitors pair price ratios with moving averages.
Z-Score Analysis: Shows Z-scores and dynamic thresholds for trading signals.
Profit and Loss (PnL): Tracks the strategy’s PnL over time.
RSI Levels: RSI with upper and lower thresholds for trade entry and exit signals.
📝 License
This project is licensed under the MIT License.
