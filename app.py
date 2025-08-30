
import pandas as pd
import numpy as np
import yfinance as yf
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go

def download_returns(ticker="AAPL", start="2018-01-01", end="2024-12-31"):
    px = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    rets = px.pct_change().dropna()
    return px, rets

def var_cvar(returns: pd.Series, alpha=0.05):
    sorted_rets = returns.sort_values()
    idx = int(alpha*len(sorted_rets))
    VaR = -sorted_rets.iloc[idx]
    CVaR = -sorted_rets.iloc[:idx].mean()
    return float(VaR), float(CVaR)

app = Dash(__name__)
app.layout = html.Div([
    html.H2("Risk Management Dashboard"),
    dcc.Input(id="ticker", value="AAPL", type="text"),
    dcc.Slider(id="alpha", min=1, max=10, value=5, step=1, marks={i:str(i/100) for i in range(1,11)}),
    dcc.Graph(id="price"),
    dcc.Graph(id="drawdown"),
    dcc.Graph(id="risk"),
])

@app.callback(
    Output("price", "figure"),
    Output("drawdown", "figure"),
    Output("risk", "figure"),
    Input("ticker", "value"),
    Input("alpha", "value"),
)
def update(ticker, alpha_int):
    alpha = alpha_int/100
    px, rets = download_returns(ticker)
    equity = (1+rets).cumprod()
    dd = equity / equity.cummax() - 1.0
    VaR, CVaR = var_cvar(rets, alpha)
    f1 = go.Figure(data=[go.Scatter(x=px.index, y=px.values, name="Price")])
    f2 = go.Figure(data=[go.Scatter(x=dd.index, y=dd.values, name="Drawdown")])
    bar = go.Bar(x=["VaR","CVaR"], y=[VaR, CVaR])
    f3 = go.Figure(data=[bar])
    return f1, f2, f3

if __name__ == "__main__":
    app.run_server(debug=False)
