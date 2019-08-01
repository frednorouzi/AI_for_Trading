# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 11:48:26 2018

@author: fred
"""
"""Plot High prices for IBM"""

import pandas as pd
import matplotlib.pyplot as plt

def test_run():
    df = pd.read_csv("../data/XOM.csv")
    # TODO: Your code here
    #print(df['High'])
    df[['High','Low', 'Close', 'Adj Close']].plot()
    plt.show()  # must be called to show plots


if __name__ == "__main__":
    test_run()

