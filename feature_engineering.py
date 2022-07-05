#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


def change_bank(df):
    temp = df.filter(items = ['offeringId', 'issuerCusip', 'offeringPricingDate', 'leftLeadFirmName']).sort_values(by = ['issuerCusip', 'offeringPricingDate'])
    temp['lagLeftLeadFirmName'] = temp['leftLeadFirmName'].shift(1)
    temp['lagissuerCusip'] = temp['issuerCusip'].shift(1)
    temp['changeBank'] = temp.apply(lambda x: True if x.leftLeadFirmName != x.lagLeftLeadFirmName and x.issuerCusip == x.lagissuerCusip else False, axis =1)
    df = df.merge(temp[['changeBank']], how = 'left', left_index = True, right_index = True)
    return df


def feature_engineering(df, test_frac = 0.2, normalize = True, random_state = 42):

    # fill na 
    df = df.fillna(0)
    
    # create new feature 
    df = change_bank(df)
    
    # split to X&y and feature selection 
    y = df.filter(like = 'post')
    X = df.loc[:, ~df.columns.isin(list(y))].drop(columns = ['offeringPricingDate', 'offeringSubSector', 'issuerCusip', 'issuerName', 'underwriters', 'leftLeadFirmId', 'leftLeadFirmName'])
    
    # train test split 
    if test_frac != 0: 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_frac, random_state=random_state)
    elif test_frac == 0:
        X_train, X_test, y_train, y_test = X, pd.DataFrame(columns = list(X)), y, pd.DataFrame(columns = list(y))
    if not normalize:  return X_train, X_test, y_train, y_test
    
    # normalize data 
    numerical_cols = list(X.select_dtypes(include=np.number))
    categorical_cols = [col for col in list(X) if col not in numerical_cols]
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    preprocessor = ColumnTransformer(
        transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)])
    X_train_transformed = preprocessor.fit_transform(X_train)
    cols = numerical_cols + list(preprocessor.named_transformers_['cat'].get_feature_names(categorical_cols))
    X_train_transformed = pd.DataFrame(X_train_transformed, columns = cols, index = X_train.index )
    if X_test.shape[0] != 0: 
        X_test_transformed = preprocessor.transform(X_test)
        X_test_transformed = pd.DataFrame(X_test_transformed, columns = cols, index = X_test.index)
    else: 
        X_test_transformed = X_test   
    return X_train_transformed, X_test_transformed, y_train, y_test
