import yfinance as yf
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def stock_retrieve(ticker,time):
    stock = yf.Ticker(ticker).history(period=time)
    return stock

def stock_analysis(ticker,help="False"):
    periods = ['1d','1mo','3mo','6mo','1y','5y','max']
    stock_data = {period: stock_retrieve(ticker, period)
                  for period in periods
                  }

    stock_returns = {period: (stock_data['1d']['Close'].iloc[-1])/stock_data[period]['Close'].iloc[0]
                     for period in periods
                     }

    stock_mean = {Period: stock_data[Period]['Close'].mean()
                  for Period in periods
                  }

    stock_vol = {Period: np.sqrt((((stock_data[Period]['Close']-stock_mean[Period])**2).sum())/(len(stock_data[Period]['Close'])-1))
                     for Period in periods[1:]
                     }
    stock_sharpe = {}
    if help=="True":
        print("[0] for stock data; [1] for stock returns; [2] for mean stock price; [3] for volatility")
    return stock_data, stock_returns, stock_mean, stock_vol
