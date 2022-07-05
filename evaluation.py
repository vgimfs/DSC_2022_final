#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import numpy as np 
import pandas as pd 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

def evaluation(true, pred):
    
    # change format of input if necessary 
    if isinstance(true, pd.DataFrame) or isinstance(true, list): 
        true = np.array(true)
    if isinstance(pred, pd.DataFrame) or isinstance(pred, list): 
        pred = np.array(pred)

    # mse 
    mse = mean_squared_error(true, pred)
    
    # direction
    n = len(true)
    true, pred = (true >= 0), (pred >= 0)
    score = 0 
    for i in range(n):
        score += accuracy_score(true[i], pred[i])
    acc = score/n
    
    return {'MSE':mse, 'ACC': acc}

