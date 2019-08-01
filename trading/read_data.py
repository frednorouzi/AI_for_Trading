# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 11:31:56 2018

@author: fred
"""
import pandas as pd


def test_run():
    """Function called by Test Run."""
    df = pd.read_csv("../data/AAPL.csv")
    # TODO: Print last 5 rows of the data frame
    print(df.tail(n=5))

if __name__ == "__main__":
    test_run()

