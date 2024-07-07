import pandas as pd
from threading import Timer
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
import yfinance as yf
import numpy as np
from scipy.stats import norm, t
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import ta
import datetime
import dash_table
from pandas.tseries.offsets import BDay
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier
import pypfopt.expected_returns as expected_returns
import pypfopt.risk_models as risk_models
import warnings
warnings.filterwarnings("ignore")

class IBApp(EWrapper, EClient):
    def __init__(self, portfolio):
        EClient.__init__(self, self)
        self.portfolio = portfolio

    def nextValidId(self, orderId):
        self.nextOrderId = orderId
        self.reqAccountUpdates(True, "")

    def updatePortfolio(self, contract: Contract, position: float,
                        marketPrice: float, marketValue: float,
                        averageCost: float, unrealizedPNL: float, realizedPNL: float,
                        accountName: str):
        self.portfolio.update_investment(contract, position, marketPrice, averageCost,marketValue,unrealizedPNL,realizedPNL)

    def accountDownloadEnd(self, accountName: str):
        # print(f"AccountDownloadEnd: {accountName}")
        self.disconnect()

    def error(self, reqId, errorCode, errorString):
        print(f"Error {reqId} {errorCode} {errorString}")


class Portfolio:
    def __init__(self):
        self.ib_app = IBApp(self)
        self.portfolio_df = pd.DataFrame(columns=["Symbol", "Position Size", "Market Price", "Average Entry Price",'Unrealized PnL','Realized PnL'])
        self.total_market_value = 0.0
        self.holidays=['2024-12-31']
    def update_investment(self, contract, position, market_price, average_cost,marketValue,unrealizedPNL,realizedPNL):
        self.total_market_value += marketValue
        new_data = {
            "Symbol": contract.symbol,
            "Position Size": np.round(position,2),
            "Market Price": np.round(market_price,3),
            "Average Entry Price": np.round(average_cost,3),
            "Market Value": np.round(marketValue,3),
            "Share of Portfolio": np.round(marketValue / self.total_market_value,3),
            "Unrealized PnL": unrealizedPNL,
            "Realized PnL": realizedPNL,

        }

        new_row = pd.DataFrame([new_data])
        self.portfolio_df = pd.concat([self.portfolio_df, new_row], ignore_index=True)

        # Update portfolio weights
        self.portfolio_df["Share of Portfolio"] = np.round(self.portfolio_df["Market Value"] / self.portfolio_df["Market Value"].sum()*100,2)
        self.portfolio_df=self.portfolio_df.sort_values(by="Share of Portfolio",ascending=False)
        self.portfolio_df=np.round(self.portfolio_df,2)

    def display_portfolio(self):
        return self.portfolio_df

    def fetch_data(self):
        self.ib_app.connect("127.0.0.1", 7497, clientId=0)
        self.ib_app.run()

    def Historical_data(self,tickers):
        prices_data = yf.download(tickers, start='2018-01-01')['Adj Close']
        prices_data=prices_data.sort_values(by='Date',ascending=False)
        prices_data.index = pd.to_datetime(prices_data.index)
        prices_data.columns.name = None
        return prices_data
    def Portfolio_mean_and_std(self,returns,weights,cov_matrix,days):
        portfolio_mean= returns.mean()@weights*days
        portfolio_std= np.sqrt(weights.T@cov_matrix@ weights)*np.sqrt(days)
        return portfolio_mean,portfolio_std
    def Returns(self):
        self.tickers=self.portfolio_df.Symbol.unique().tolist()
        weights=np.array(self.portfolio_df["Share of Portfolio"])/100
        prices_data=self.Historical_data(self.tickers)
        prices_data=prices_data[252:]
        prices_data=prices_data.dropna()
        returns=np.log(prices_data/prices_data.shift(-1))
        returns=returns.dropna()
        cov_matrix= returns.cov()
        Portfolio_Returns=returns.copy()
        portfolio_mean,portfolio_std= self.Portfolio_mean_and_std(Portfolio_Returns,weights,cov_matrix,1)
        
        return portfolio_mean,portfolio_std,returns.dot(weights)
    
    def historicalVaR(self,returns, alpha=5,days=1):
        return -np.percentile(returns, alpha)*np.sqrt(days)
    
    @staticmethod
    def var_parametric(portofolioReturns, portfolioStd, distribution='normal', alpha=0.95, dof=6,days=1):
        if distribution == 'normal':
            VaR = norm.ppf(alpha, portofolioReturns,portfolioStd)*np.sqrt(days)
        elif distribution == 't':
            VaR = t.ppf(alpha, dof, portofolioReturns, portfolioStd)*np.sqrt(days)
        return VaR

    def Portfolio_Risk(self):
        portfolio_mean, portfolio_std,hist_returns = self.Returns()
        Historical_VaR=self.historicalVaR(hist_returns)*self.total_market_value
        Parametric_VaR=self.var_parametric(portfolio_mean,portfolio_std)*self.total_market_value
        return Historical_VaR,Parametric_VaR
    
    def Watch_List(self):
        data=pd.read_csv("Price_Data.csv")
        latest_date=pd.to_datetime(data.Date[0]).date()
        print(latest_date)
        if latest_date==(datetime.datetime.today()-BDay(1)).date():
            print("Data for Watchlist is already Pulled")
            prices_data=data.copy()
            prices_data=prices_data.set_index("Date")
        else:
            print("Pulling Watchlist data")
            prices_data=yf.download(data.columns[1:], start=latest_date)[-252:]['Adj Close']
            merged=pd.concat([data.set_index('Date'),prices_data],axis=0)
            def parse_date(date_str):
                for fmt in ('%Y-%m-%d', '%d-%m-%Y', '%d/%m/%Y %H:%M'):
                    try:
                        return pd.to_datetime(date_str, format=fmt)
                    except ValueError:
                        continue
                raise ValueError(f"Date format not recognized for: {date_str}")

            merged.index = merged.index.to_series().apply(parse_date)
            merged.index = merged.index.strftime('%Y-%m-%d')
            merged = merged[~merged.index.duplicated(keep='last')]
            merged=merged.sort_index(ascending=False)
            merged.to_csv("Price_Data.csv")
            prices_data=merged.copy()

        start_date = (datetime.datetime.today() - pd.DateOffset(months=6)).date()
        prices_data.index=pd.to_datetime(prices_data.index)
        prices_train = prices_data.loc[start_date:]
        df = prices_train.copy()
        df = df.loc[:, ~(df < 6).any()]
        df = df.dropna(axis=1)
        mu = expected_returns.mean_historical_return(df)
        S = risk_models.risk_matrix(df, method='ledoit_wolf')
        info=pd.read_excel('Stock List.xlsx')
        sector_info = info.set_index('Ticker')['Sector'].to_dict()
        sector_max_allocation = {sector: 0.2 for sector in set(sector_info.values())}
        sector_map = {sector: [] for sector in sector_max_allocation.keys()}
        for ticker in df.columns:
            sector = sector_info.get(ticker, None)
            if sector:
                sector_map[sector].append(ticker)

        sector_map = {sector: [] for sector in sector_max_allocation.keys()}
        for ticker in df.columns:
            sector = sector_info.get(ticker, None)
            if sector:
                sector_map[sector].append(ticker)

        # Efficient Frontier
        ef = EfficientFrontier(mu, S, weight_bounds=(0, 0.4))

        # Add sector constraints directly
        for sector, tickers in sector_map.items():
            indices = [df.columns.get_loc(ticker) for ticker in tickers if ticker in df.columns]
            if indices:
                ef.add_constraint(lambda w, indices=indices: sum(w[i] for i in indices) <= sector_max_allocation[sector])

        # Optimize portfolio for max Sharpe ratio
        # raw_weights = ef.max_sharpe()
        i = 0
        while True:
            try:
                raw_weights = ef.efficient_return(target_return=mu.max()-i)
                print(f"Selected i is {i}")
                break
            except Exception as e:
                # print(f"Error for i={i}: {e}")
                i += 0.01
        cleaned_weights = ef.clean_weights()
        ef.portfolio_performance(verbose=True)
        filtered_data = [(name, value) for name, value in cleaned_weights.items() if value > 0.01]
        watch_list=pd.DataFrame(filtered_data)
        watch_list.columns=['Stock','Weight']
        print(watch_list)
        return watch_list
    
    def Portfolio_log(self):
        data=self.portfolio_df
        print(data)
        Historical_VaR,Parametric_VaR=self.Portfolio_Risk()
        log=pd.read_excel("Portfolio_Log.xlsx")
        daily_log={
            "Date":datetime.datetime.today().date(),
            "Market Value": np.round(data['Market Value'].sum(),3),
            "Average Entry Price": np.round((data["Average Entry Price"]*data['Position Size']).sum(),3),
            "Unrealized PnL":np.round(data["Unrealized PnL"].sum(),3),
            "Realized PnL":np.round(data["Realized PnL"].sum(),3),
            "Historical_VaR":Historical_VaR,
            "Parametric_VaR":Parametric_VaR
            }

        daily_log = pd.DataFrame([daily_log])
        summary=pd.concat([log,daily_log],axis=0)
        summary['Δ Unrealized']=np.round(summary['Unrealized PnL']-summary['Unrealized PnL'].shift(1),2)
        summary['Δ Realized']=np.round(summary['Realized PnL']-summary['Realized PnL'].shift(1),2)
        summary['Date']=pd.to_datetime(summary['Date'])
        summary=summary.drop_duplicates(subset='Date')
        summary['Date']=summary['Date'].dt.strftime('%Y-%m-%d')
        summary.iloc[:,1:]=np.round(summary.iloc[:,1:],2)
        summary.to_excel("Portfolio_Log.xlsx",index=False)
        return summary
    
    def Technical_indicators(self):
        self.tickers=self.portfolio_df.Symbol.unique().tolist()
        prices_data=self.Historical_data(self.tickers)
        prices_data=prices_data.sort_index()
        for i in prices_data.columns:
            prices_data['ema_7_'+i] = prices_data[i].ewm(span=7, adjust=False).mean()
            prices_data['ema_14_'+i] = prices_data[i].ewm(span=14, adjust=False).mean()
            prices_data['SMA_7_'+i] = ta.trend.sma_indicator(prices_data[i], window=7)
            prices_data['SMA_14_'+i] = ta.trend.sma_indicator(prices_data[i], window=14)
            prices_data['SMA_50_'+i] = ta.trend.sma_indicator(prices_data[i], window=50)
            prices_data['SMA_200_'+i] = ta.trend.sma_indicator(prices_data[i], window=200)

        prices_data=prices_data.dropna(axis=0)

        return prices_data
    
    def Plots(self):
        prices_data = self.Technical_indicators()
        figures = []
        for ticker in self.tickers:
            fig = go.Figure()
            indicators = [col for col in prices_data.columns if ticker in col]
            for indicator in indicators:
                fig.add_trace(go.Scatter(x=prices_data.index, y=prices_data[indicator], mode='lines', name=indicator))
            fig.update_layout(title=f'Technical Indicators for {ticker}', xaxis_title='Date', yaxis_title='Price')
            figures.append(fig)
        return figures
    

def create_dashboard(app, portfolio):
    app.layout = dbc.Container([
        dbc.Navbar(
            dbc.Container([
                dbc.NavbarBrand("Portfolio Dashboard", className="mx-auto")
            ]),
            color="dark",
            dark=True,
        ),

        html.Br(),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H2("Current Portfolio")),
                    dbc.CardBody(dash_table.DataTable(
                        id='current-portfolio',
                        columns=[{"name": i, "id": i} for i in portfolio.display_portfolio().columns],
                        data=portfolio.display_portfolio().to_dict('records') + [{
                            'Symbol': 'Grand Total',
                            'Position Size': '',
                            'Market Price': '',
                            'Average Entry Price': '',
                            'Market Value': np.round(portfolio.display_portfolio()['Market Value'].sum(),2),
                            'Share of Portfolio': np.round(portfolio.display_portfolio()['Share of Portfolio'].sum(),2),
                            'Unrealized PnL': np.round(portfolio.display_portfolio()['Unrealized PnL'].sum(),2),
                            "Realized PnL": np.round(portfolio.display_portfolio()['Realized PnL'].sum(),2)}],

                        filter_action="native",
                        sort_action="native",
                        sort_mode="multi",
                        page_size=20,
                        style_table={'overflowX': 'auto'},
                        style_header={
                            'backgroundColor': '#272727',
                            'fontWeight': 'bold',
                            'color': 'white'
                        },
                        style_data_conditional=[
                            {
                                'if': {'filter_query': '{symbol} = "Grand Total"'},
                                'backgroundColor': '#d9edf7',
                                'fontWeight': 'bold'
                            }
                        ],
                        style_cell={
                            'backgroundColor': 'white',
                            'color': 'black',
                            'textAlign': 'center'
                        }
                    ))
                ], className="mb-4")
            ], width=12),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H2("Portfolio Log")),
                    dbc.CardBody(dash_table.DataTable(
                        id='portfolio-log',
                        columns=[{"name": i, "id": i} for i in portfolio.Portfolio_log().columns],
                        data=portfolio.Portfolio_log().to_dict('records'),
                        filter_action="native",
                        sort_action="native",
                        sort_mode="multi",
                        page_size=20,
                        style_table={'overflowX': 'auto'},
                        style_header={
                            'backgroundColor': '#272727',
                            'fontWeight': 'bold',
                            'color': 'white'
                        },
                        style_cell={
                            'backgroundColor': 'white',
                            'color': 'black',
                            'textAlign': 'center'
                        }
                    ))
                ], className="mb-4")
            ], width=12),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H3("Watchlist")),
                    dbc.CardBody(dash_table.DataTable(
                        id='watchlist',
                        columns=[{"name": i, "id": i} for i in portfolio.Watch_List().columns],
                        data=portfolio.Watch_List().to_dict('records'),
                        filter_action="native",
                        sort_action="native",
                        sort_mode="multi",
                        page_size=20,
                        style_table={'overflowX': 'auto'},
                        style_header={
                            'backgroundColor': '#272727',
                            'fontWeight': 'bold',
                            'color': 'white'
                        },
                        style_cell={
                            'backgroundColor': 'white',
                            'color': 'black',
                            'textAlign': 'center'
                        }
                    ))
                ], className="mb-4")
            ], width=12)
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H2("Technical Indicators")),
                    dbc.CardBody([
                        dcc.Dropdown(id='ticker-dropdown', multi=True, options=[
                            {'label': ticker, 'value': ticker} for ticker in portfolio.portfolio_df["Symbol"].unique()
                        ], placeholder="Select Tickers", style={"margin-bottom": "15px"}),
                        dcc.Graph(id='indicator-plot')
                    ])
                ])
            ], width=12)
        ])
    ], fluid=True)

    @app.callback(
        Output('indicator-plot', 'figure'),
        [Input('ticker-dropdown', 'value')]
    )
    def update_plots(selected_tickers):
        if not selected_tickers:
            return go.Figure()
        figures = portfolio.Plots()
        fig = go.Figure()
        for ticker in selected_tickers:
            for ticker_fig in figures:
                if ticker in ticker_fig['layout']['title']['text']:
                    for trace in ticker_fig['data']:
                        fig.add_trace(trace)
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        return fig

def main():
    portfolio = Portfolio()
    portfolio.fetch_data()

    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
    create_dashboard(app, portfolio)
    app.run_server(debug=True)

if __name__ == "__main__":
    main()
        
