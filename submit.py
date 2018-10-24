
# coding: utf-8

# In[33]:


import pandas as pd
import numpy as np
from load import load_ETF
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn import svm

import math
import os

from class_onedaymodel import normalize_data,denormalize,sign,LinearRegressionReg

from load import load_ETF



#load data
def load_alldata_feature(stock, seq_len,feature,day=0):
    amount_of_features = len(stock.columns) 
    data = stock.as_matrix() 
    sequence_length = seq_len + 1 # index starting from 0
    result = []
    
    for index in range(len(data) - sequence_length): # maxmimum date = lastest date - sequence length
        result.append(data[index: index + sequence_length]) # index : index + 21days
    
    result = np.array(result)
    row = result.shape[0]-1 
#     train = result[:int(row), :] # 90% date, all features 
    
    x_train = result[:, :-1] 
    y_train = result[:,-1,feature] 

    x_test = result[-1,1:] 
#     y_test = result[int(row):, -1,feature]

    
    x_train = x_train.reshape((x_train.shape[0],x_train.shape[1]*x_train.shape[2]))
    x_test =  x_test.reshape((x_test.shape[0]*x_test.shape[1])) #shape(127,20,6) --> (127,120)
     

    return [x_train, y_train , x_test]

def get_fiveday_close(stock,stock_de,seq_len,lamb):
    amount_of_features = len(stock.columns)
    models = []
    features = []
    columns = stock.columns
    close_price = []
    
    for i in range(amount_of_features): #load data and train
        x_train,y_train,x_test = load_alldata_feature(stock,seq_len,i)
        
        lr = LinearRegressionReg()
        lr.fit(x_train,y_train,lamb)
        models.append(lr)
        features.append(lr.predict(x_test))
    
    lastday_close = denormalize(stock_de,x_test[-1],'close').reshape((1))
    features_ = np.array(features)    
    x_test = np.concatenate((x_test[5:],features_),axis = 0)
    close_price.append(features_[-1])
    
    for day in range(4):
        for feature in range(amount_of_features):
            features_[feature] = models[feature].predict(x_test)
        x_test = np.concatenate((x_test[5:],features_),axis = 0)
        close_price.append(features_[-1])
    
    close_price = np.array(close_price)
    close_price = denormalize(stock_de,close_price,'close')
    return close_price, lastday_close

    

def predict_ETF_close(ETFtable,ETF_list,window,lamb):
    ETF_week_close = []
    ETF_week_score = []
    ETF_lastday_close = []
    for ETF_code in ETF_list:
        ETF_target = load_ETF(ETFtable,ETF_code)
        ETF_target_N = normalize_data(ETF_target)
        close_price , lastday_close = get_fiveday_close(ETF_target_N,ETF_target,window,lamb)
        
        ETF_week_close.append(close_price)
        ETF_lastday_close.append(lastday_close)
        
    ETF_week_close = np.array(ETF_week_close)
    ETF_lastday_close = np.array(ETF_lastday_close)
    
    return ETF_week_close,ETF_lastday_close


def submit_data(ETFtable,ETF_list,window,lamb): 
    ETF_week_close,ETF_last_close = predict_ETF_close(ETFtable,ETF_list,window,lamb)

    ETF_week = []
    ETF_week_close = ETF_week_close.reshape((18,5))
    ETF_week_close_ = np.concatenate((ETF_last_close,ETF_week_close),axis = 1)

    for code,price_list in enumerate(ETF_week_close_):
        temp = []
        for day,price in enumerate(price_list):
            if(day != 0):
                temp.append(sign(price_list[day]-price_list[day-1]))
                temp.append(price)
        ETF_week.append(temp)


    ETF_week_df = pd.DataFrame(ETF_week,index = ETF_list,columns =['Mon_ud','Mon_cprice','Tue_ud','Tue_cprice','Wed_ud','Wed_cprice','Thu_ud',
                                                 'Thu_cprice','Fri_ud','Fri_cprice'])
    ETF_week_df.to_csv(os.path.join(os.getcwd(),'Submission.csv'))
    print('Done')




