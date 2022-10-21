# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 14:57:43 2022

@author: Naresh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori
market_data = pd.read_csv(r'Market_Basket_Optimisation.csv', header = None)
transacts = []
for i in range(0, len(market_data)):
  transacts.append([str(market_data.values[i,j]) for j in range(0, 20)])
  rules = apriori(transactions = transacts, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)
  def inspect(output):
    Left_Hand_Side = [tuple(result[2][0][0])[0] for result in output]
    support = [result[1] for result in output]
    confidence = [result[2][0][2] for result in output]
    lift = [result[2][0][3] for result in output]
    Right_Hand_Side = [tuple(result[2][0][1])[0] for result in output]
    return list(zip(Left_Hand_Side, support, confidence, lift, Right_Hand_Side))

output = list(rules)
output_data = pd.DataFrame(inspect(output), columns = ['Left_Hand_Side', 'Support', 'Confidence', 'Lift', 'Right_Hand_Side'])
print(output_data)
print(output_data.nlargest(n = 5, columns = 'Lift'))
