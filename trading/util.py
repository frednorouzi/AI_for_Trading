# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 12:25:25 2018

@author: fred
"""
import os
import pandas as pd

def symbol_to_path(symbol, base_dir="data"):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))


def get_data(symbols, dates):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    if 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols.insert(0, 'SPY')

    for symbol in symbols:
        # TODO: Read and join data for each symbol
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col = 'Date',
        parse_dates = True, usecols=['Date', 'Adj Close'], na_values=['nan'])
        df = df.join(df_temp)
        if symbol == 'SPY': #drop dates SPY did not trade
            df = df.dropna(subset = ['SPY'])
    return df


def test_run():
    # Define a date range
    dates = pd.date_range('2016-01-22', '2016-01-26')

    # Choose stock symbols to read
    symbols = ['GOOGL', 'XOM', 'GLD']
    
    # Get stock data
    df = get_data(symbols, dates)
    print(df)

