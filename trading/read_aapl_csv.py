# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 10:16:03 2016

@author: Fred
"""
import pandas as pd
def test_run():
     df = pd.read_csv('data/AAPL.csv')
     print df
     
if __name__ == "__main__":
    test_run()
     