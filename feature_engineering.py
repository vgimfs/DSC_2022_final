#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
def feature_engineering(df):
    '''given the cmg data, we want to return '''
    
    df = df.apply(lambda x: x.fillna(value=df['filingDetailsOfferingPrice']))
    
    # how many offerings that have a change from the previous bank
    temp = df.filter(items = ['offeringId', 'issuerCusip', 'offeringPricingDate', 'leftLeadFirmName']).sort_values(by = ['issuerCusip', 'offeringPricingDate'])
    temp['lagLeftLeadFirmName'] = temp['leftLeadFirmName'].shift(1)
    temp['lagissuerCusip'] = temp['issuerCusip'].shift(1)
    temp['changeBank'] = temp.apply(lambda x: True if x.leftLeadFirmName != x.lagLeftLeadFirmName and x.issuerCusip == x.lagissuerCusip else False, axis =1)
    df = df.merge(temp[['changeBank']], how = 'left', left_index = True, right_index = True)
    
    y = df[['1SharePrice', '7SharePrice', '30SharePrice', '90SharePrice', '180SharePrice']]
    X = df.loc[:, ~df.columns.isin(list(y))].drop(columns = ['offeringPricingDate', 'offeringsSubSector', 'issuerCusip', 'issuerName', 'underwriters', 'leftLeadFirmId', 'leftLeadFirmName'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    
    numerical_cols = list(X.select_dtypes(include=np.number))
    categorical_cols = [col for col in list(X) if col not in numerical_cols]
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    preprocessor = ColumnTransformer(
        transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)])
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.fit_transform(X_test)
    cols = numerical_cols + list(preprocessor.named_transformers_['cat'].get_feature_names(categorical_cols))
    X_train_transformed = pd.DataFrame(X_train_transformed, columns = cols, index = X_train.index )
    X_test_transformed = pd.DataFrame(X_test_transformed, columns = cols, index = X_test.index)
    
    return X_train_transformed, X_test_transformed, y_train, y_test



