import pandas as pd
import numpy as np
import yfinance as yf
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from datetime import datetime
import pytz
import warnings
import ta
warnings.filterwarnings("ignore")
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_table
import plotly.graph_objs as go

class WatchList:
    def __init__(self):
        self.current_file = pd.read_csv("Price_Data.csv")
        self.stock_info = pd.read_excel("Stock List.xlsx")
        self.latest_pull_date = self.current_file.Date.iloc[-1]
        self.current_us_time = datetime.now(pytz.timezone('US/Eastern'))
        self.stock_to_pull = self.current_file.columns[1:].tolist()  # Convert Index to list
        self.range_to_analyse = (datetime.today() - pd.DateOffset(months=6)).date()
        self.start_date_for_technical_indicators = (datetime.today() - pd.DateOffset(months=24)).date()
        self.date_today = str(datetime.today().date())

    def yahoo_pull(self, stocks, date):
        return yf.download(stocks, start=date)['Adj Close']

    def pull_data(self):
        if pd.to_datetime(self.latest_pull_date).date() <= self.current_us_time.date():
            print("Data is already pulled for today")
            prices_data = self.current_file.copy()
        else:
            print("Pulling data")
            prices_data = self.yahoo_pull(self.stock_to_pull, pd.to_datetime(self.latest_pull_date).date())
        return prices_data

    def data_parsing(self):
        data = self.pull_data()
        remove_stock = ['AEHL', 'EBS', 'WGS', 'SMR', 'ROOT', 'ANF', 'INSG']
        data = data.drop(columns=remove_stock)
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.set_index("Date")
        data = data.loc[self.range_to_analyse:]
        data = data.loc[:, ~(data < 1).any()]
        return data

    def mean_and_std(self, df):
        mu = expected_returns.mean_historical_return(df)
        S = risk_models.risk_matrix(df, method='ledoit_wolf')
        return mu, S

    def set_sector_constraints(self, df):
        sector_info = self.stock_info.set_index('Ticker')['Sector'].to_dict()
        sector_max_allocation = {sector: 0.2 for sector in set(sector_info.values())}
        sector_map = {sector: [] for sector in sector_max_allocation.keys()}
        for ticker in df.columns:
            sector = sector_info.get(ticker, None)
            if sector:
                sector_map[sector].append(ticker)
        return sector_max_allocation, sector_map

    def optimized_portfolio_maximal_return(self, df):
        mu, S = self.mean_and_std(df)
        sector_max_allocation, sector_map = self.set_sector_constraints(df)
        ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
        for sector, tickers in sector_map.items():
            indices = [df.columns.get_loc(ticker) for ticker in tickers if ticker in df.columns]
            if indices:
                ef.add_constraint(lambda w, indices=indices: sum(w[i] for i in indices) <= sector_max_allocation[sector])
        i = 0
        while True:
            try:
                raw_weights = ef.efficient_return(target_return=mu.max() - i)
                print(f"Selected i is {i}")
                break
            except Exception as e:
                i += 0.01
        cleaned_weights = ef.clean_weights()
        ef.portfolio_performance(verbose=True)
        filtered_data = pd.DataFrame([(name, value) for name, value in cleaned_weights.items() if value > 0.01])
        filtered_data.columns = ['Symbol', 'Weight']
        filtered_data.to_hdf(f'filtered_df_{self.date_today}.h5', key='df', mode='w')
        return filtered_data

    def optimized_portfolio_max_sharpe(self, df):
        mu, S = self.mean_and_std(df)
        try:
            ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
            raw_weights = ef.max_sharpe()
            cleaned_weights = ef.clean_weights()
            ef.portfolio_performance(verbose=True)
            filtered_data = pd.DataFrame([(name, value) for name, value in cleaned_weights.items() if value > 0.01])
            filtered_data.columns = ['Symbol', 'Weight']
            filtered_data.to_hdf(f'filtered_df_{self.date_today}.h5', key='df', mode='w')
            return filtered_data
        except Exception as e:
            print(f"Optimization failed: {e}")
            return None

    def technical_indicators(self, df, type_of_optimizations='Max_Return'):
        if type_of_optimizations == 'Max_Return':
            self.chosen_stocks = self.optimized_portfolio_maximal_return(df).Symbol.unique().tolist()
        else:
            self.chosen_stocks = self.optimized_portfolio_max_sharpe(df).Symbol.unique().tolist()

        prices_data = self.yahoo_pull(self.chosen_stocks, self.start_date_for_technical_indicators)
        prices_data = prices_data.sort_index()
        for i in prices_data.columns:
            prices_data['ema_7_' + i] = prices_data[i].ewm(span=7, adjust=False).mean()
            prices_data['ema_14_' + i] = prices_data[i].ewm(span=14, adjust=False).mean()
            prices_data['SMA_7_' + i] = ta.trend.sma_indicator(prices_data[i], window=7)
            prices_data['SMA_14_' + i] = ta.trend.sma_indicator(prices_data[i], window=14)
            prices_data['SMA_50_' + i] = ta.trend.sma_indicator(prices_data[i], window=50)
            prices_data['SMA_200_' + i] = ta.trend.sma_indicator(prices_data[i], window=200)
        return prices_data, self.chosen_stocks

    def plots(self, df):
        prices_data, chosen_stocks = self.technical_indicators(df)
        figures = []
        for ticker in chosen_stocks:
            fig = go.Figure()
            indicators = [col for col in prices_data.columns if ticker in col]
            for indicator in indicators:
                fig.add_trace(go.Scatter(x=prices_data.index, y=prices_data[indicator], mode='lines', name=indicator))
            fig.update_layout(title=f'Technical Indicators for {ticker}', xaxis_title='Date', yaxis_title='Price')
            figures.append(fig)
        return figures

app = dash.Dash(__name__)
watch_list = WatchList()
filtered_df = watch_list.optimized_portfolio_maximal_return(watch_list.data_parsing())
figures = watch_list.plots(watch_list.data_parsing())

app.layout = html.Div([
    html.H1("Optimized Portfolio and Technical Indicators"),
    html.H2("Filtered DataFrame"),
    dash_table.DataTable(
        id='filtered-table',
        columns=[{"name": i, "id": i} for i in filtered_df.columns],
        data=filtered_df.to_dict('records'),
        style_table={'overflowX': 'auto'}
    ),
    html.H2("Technical Indicator Plots"),
    dcc.Graph(id='indicator-plot', figure=figures[0]),
    dcc.Dropdown(
        id='ticker-dropdown',
        options=[{'label': ticker, 'value': i} for i, ticker in enumerate(watch_list.chosen_stocks)],
        value=0  # Default to the first ticker
    )
])

@app.callback(
    Output('indicator-plot', 'figure'),
    [Input('ticker-dropdown', 'value')]
)
def update_plot(selected_index):
    return figures[selected_index]

if __name__ == '__main__':
    app.run_server(debug=True)
