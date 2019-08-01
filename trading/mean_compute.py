# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 11:37:05 2018

@author: fred
"""
"""Compute mean volume"""

import pandas as pd

def get_mean_volume(symbol):
    """Return the mean volume for stock indicated by symbol.
    
    Note: Data for a stock is stored in file: data/<symbol>.csv
    """
    df = pd.read_csv("../data/{}.csv".format(symbol))  # read in data
    # TODO: Compute and return the mean volume for this stock
    return df["Adj Close"].mean()

def test_run():
    """Function called by Test Run."""
    print("Mean Adj Close")
    for symbol in ['AAPL', 'GLD', 'googl', 'XOM', 'SPY']:
        print( '{0:}:{1:>2,.4f}'.format(symbol, get_mean_volume(symbol)))


if __name__ == "__main__":
    test_run()

