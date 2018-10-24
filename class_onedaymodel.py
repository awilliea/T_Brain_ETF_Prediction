
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from load import load_ETF
import matplotlib.pyplot as plt

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

from sklearn import preprocessing
from sklearn import svm

import math


def get_change_rate(stock,back_day = 1):
    stock_r = stock.replace(0,0.1) 
    for back in range(back_day):
        if(back == 0):
            stock_rate = (stock_r - stock_r.shift(back+1).fillna(0))/stock_r
        else:
            stock_temp = (stock_r - stock_r.shift(back+1).fillna(0))/stock_r
            
            #rename columns
            column_names = list(stock_temp.columns)
            dict_ = {}
            for column_name in column_names:
                dict_[column_name] = column_name+f'_{back}'
                
            stock_temp = stock_temp.rename(columns = dict_)
            
            stock_rate = pd.concat([stock_rate,stock_temp],axis = 1)
    
    return stock_rate[back_day:]

def rise_fall(rate,describe,interval): #
    mean = describe[1]
    std = describe[2]
    if (rate >= mean+std*interval):
        return 1
    elif(rate <= mean-std*interval):
        return -1
    else :
        return 0

def sign(x):
    if (x > 0):
        return 1
    elif(x < 0):
        return -1
    else:
        return 0

def get_rise_fall(df,interval):
    df_rise_fall_d = df['close'].describe()
#     df_i_rise_fall_d = df['i_close'].describe()
    
    trf = []
    rf = []
    irf = []
    for day in range(df.shape[0]):
        rf.append(rise_fall(df['close'][day],df_rise_fall_d,interval))
#         irf.append(rise_fall(df['i_close'][day],df_i_rise_fall_d))
        trf.append(sign(df['close'][day]))
#     df['i_rise_fall'] = irf 
    df['rise_fall'] = rf
    df['t_rise_fall'] = trf
    
#data type
def normalize_data(df):
    min_max_scaler = preprocessing.MinMaxScaler()
    df_N = df.copy()
    
    for columnname in df_N.columns:
        df_N[columnname] = min_max_scaler.fit_transform(df_N[columnname].values.reshape(-1,1))
    
    return df_N 

def denormalize(stock, normalized_value,columnname): 
    stock_price = stock[columnname].values.reshape(-1,1)
    normalized_value = normalized_value.reshape(-1,1)

    #return df.shape, p.shape
    min_max_scaler = preprocessing.MinMaxScaler()
    stock_price_N = min_max_scaler.fit_transform(stock_price)
    new = min_max_scaler.inverse_transform(normalized_value)
    return new

#load data
def load_data(stock, seq_len):
    amount_of_features = len(stock.columns) 
    data = stock.as_matrix() 
    sequence_length = seq_len + 1 # index starting from 0
    result = []
    
    for index in range(len(data) - sequence_length): # maxmimum date = lastest date - sequence length
        result.append(data[index: index + sequence_length]) # index : index + 20days
    
    result = np.array(result)
    row = round(0.9 * result.shape[0]) # 90% split
    train = result[:int(row), :] # 90% date, all features 
    
    x_train = train[:, :-1] #make the last day of train data as y_train
    y_train = train[:, -1][:,-1] #the close price of the last day of every data 
    
    x_test = result[int(row):, :-1] 
    y_test = result[int(row):, -1][:,-1]

    x_train = x_train.reshape((x_train.shape[0],x_train.shape[1]*x_train.shape[2]))
    x_test =  x_test.reshape((x_test.shape[0],x_test.shape[1]*x_test.shape[2]))
     

    return [x_train, y_train, x_test, y_test]

#load data
def load_data_regression(stock, seq_len):
    t = stock['close'].values
    stock_r = stock.drop(['close'],axis=1)
    stock_r['close'] = t
    
    amount_of_features = len(stock.columns) 
    data = stock.as_matrix() 
    sequence_length = seq_len + 1 # index starting from 0
    result = []
    
    for index in range(len(data) - sequence_length): # maxmimum date = lastest date - sequence length
        result.append(data[index: index + sequence_length]) # index : index + 20days
    
    result = np.array(result)
    row = round(0.9 * result.shape[0]) # 90% split
    train = result[:int(row), :] # 90% date, all features 
    
    x_train = train[:, :-1] #not use close and rise_fall 
    y_train = train[:, -1][:,-1] #the close price of the last day of every data 
    
    x_test = result[int(row):, :-1] 
    y_test = result[int(row):, -1][:,-1]

    x_train = x_train.reshape((x_train.shape[0],x_train.shape[1]*x_train.shape[2]))
    x_test =  x_test.reshape((x_test.shape[0],x_test.shape[1]*x_test.shape[2]))
     

    return [x_train, y_train, x_test, y_test]

# Model
class LinearRegressionReg:

    def __init__(self):
        self._dimension = 0

    def fit(self, X, Y, lamb):  #calculate w
        self._dimension = X.shape[1]
        self._w = np.zeros((self._dimension,1))
        self._lamb = lamb
        self._w = np.linalg.inv(np.dot(X.T, X) + lamb*np.eye(self._dimension)).dot(X.T).dot(Y)

    def predict(self, X):
        result = np.dot(X, self._w)
        return result
    
    def fiveday_predict(self,X,Y):
        for i in range(5):
            Y_predict = self.predict(X)
    
    def error(self, X, Y):  #squared error
        Y_predict = self.predict(X)
        return sum((Y_predict-Y)**2)/(len(Y)*1.0)            

    def get_w(self):
        return self._w

    def print_val(self):
        print ("w: ", self._w)
    
    def score(self,X,Y,original_stock):
        Y_predict = self.predict(X)
        Y_predict_de = denormalize(original_stock,Y_predict,'close').reshape((Y.shape[0]))
        Y_test_de = denormalize(original_stock,Y,'close').reshape((Y.shape[0]))
        return sum(((Y_test_de-abs(Y_predict_de-Y_test_de))/Y_test_de)*0.5)/(len(Y_test_de)*1.0)


# predict


def one_day_prediction_regression_18(ETFtable,ETF_list,window,lamb,interval):
    ETF_score = []
    for ETF_code in ETF_list:
        ETF_target = load_ETF(ETFtable,ETF_code)
        
        #price data
        ETF_target_N = normalize_data(ETF_target)
        X_train_p, y_train_p, X_test_p, y_test_p = load_data(ETF_target_N, window)
        
        #rate data
        ETF_target_rate = get_change_rate(ETF_target)
        get_rise_fall(ETF_target_rate,interval)
        X_train_r, y_train_r, X_test_r, y_test_r = load_data(ETF_target_rate, window)
        
        ### Model
        lr = LinearRegressionReg()
        lr.fit(X_train_p, y_train_p, lamb)

        ### score
        p = lr.predict(X_test_p)
        y_predict_de = denormalize(ETF_target,p,'close').reshape((y_test_p.shape[0]))
        
        y_predict_rise_fall = []
        for day,price in enumerate(y_predict_de):
                if (day != 0):
                    y_predict_rise_fall.append(sign(y_predict_de[day]-y_predict_de[day-1]))                
        y_predict_rise_fall = np.array(y_predict_rise_fall).astype(np.float64)
        
        if(y_test_r.shape[0] > y_predict_rise_fall.shape[0]):
            y_test_r_short = y_test_r[y_test_r.shape[0]-y_predict_rise_fall.shape[0]:] 
            score = (sum(y_test_r_short == y_predict_rise_fall)*0.5)/(y_test_r_short.shape[0]*1.0)
        else:
            score = (sum(y_test_r == y_predict_rise_fall)*0.5)/(y_test_r.shape[0]*1.0)

        print(f'{ETF_code} average one day score is',score)
        ETF_score.append(score)
        
    print('Total ETF one day rise_fall score is',sum(ETF_score))    
    print('Average ETF one day rise_fall score is',sum(ETF_score)/18.0)
    return ETF_score




def one_day_prediction_regression(ETF,window,lamb,interval):
    #price data
    ETF_target_N = normalize_data(ETF)
    X_train_p, y_train_p, X_test_p, y_test_p = load_data(ETF_target_N, window)

    #rate data
    ETF_target_rate = get_change_rate(ETF)
    get_rise_fall(ETF_target_rate,interval)
    X_train_r, y_train_r, X_test_r, y_test_r = load_data(ETF_target_rate, window)

    ### Model
    lr = LinearRegressionReg()
    lr.fit(X_train_p, y_train_p, lamb)

    ### score
    p = lr.predict(X_test_p)
    y_predict_de = denormalize(ETF,p,'close').reshape((y_test_p.shape[0]))

    y_predict_rise_fall = []
    for day,price in enumerate(y_predict_de):
            if (day != 0):
                y_predict_rise_fall.append(sign(y_predict_de[day]-y_predict_de[day-1]))                
    y_predict_rise_fall = np.array(y_predict_rise_fall).astype(np.float64)

    if(y_test_r.shape[0] > y_predict_rise_fall.shape[0]):
        y_test_r_short = y_test_r[y_test_r.shape[0]-y_predict_rise_fall.shape[0]:] 
        score = (sum(y_test_r_short == y_predict_rise_fall)*0.5)/(y_test_r_short.shape[0]*1.0)
    else:
        score = (sum(y_test_r == y_predict_rise_fall)*0.5)/(y_test_r.shape[0]*1.0)

    print('Average one day score is',score)
    




def one_day_prediction_svm_18(ETFtable ,ETF_list,window,interval,kernel = 'rbf',gamma = 1,C =1.0,coef0 = 0 ):
    ETF_score = []
    for ETF_code in ETF_list:
        ETF_target = load_ETF(ETFtable,ETF_code)
        ETF_target_rate = get_change_rate(ETF_target)
        get_rise_fall(ETF_target_rate,interval)
        X_train, y_train, X_test, y_test = load_data(ETF_target_rate, window)

        # Model
        clf = svm.SVC(kernel=kernel,gamma = gamma,C=C,coef0=coef0)
        clf.fit(X_train,y_train)
        
        y_predict_test = clf.predict(X_test)
        rbf_test = sum(y_predict_test == y_test)*0.5/(y_test.shape[0]*1.0)
        
        y_predict_train = clf.predict(X_train)
        rbf_train = sum(y_predict_train == y_train)*0.5/(y_train.shape[0]*1.0)
#         print('Average train one day score is ',sum(y_predict_train == y_train)*0.5/(y_train.shape[0]*1.0))
#         print('Average test one day score is ',sum(y_predict_test == y_test)*0.5/(y_test.shape[0]*1.0))

        print(f'{ETF_code} average one day score is ,',rbf_test)
        ETF_score.append(rbf_test)

    print('Total ETF one day rise_fall score is',sum(ETF_score))    
    print('Average ETF one day rise_fall score is',sum(ETF_score)/18.0)




def one_day_prediction_svm(ETF,window,interval,kernel = 'rbf',gamma = 1,C =1.0,coef0 = 0 ):
    ETF_target_rate = get_change_rate(ETF)
    get_rise_fall(ETF_target_rate,interval)
    X_train, y_train, X_test, y_test = load_data(ETF_target_rate, window)

    # Model
    clf = svm.SVC(kernel=kernel,gamma = gamma,C=C,coef0=coef0)
    clf.fit(X_train,y_train)

    y_predict_test = clf.predict(X_test)
    rbf_test = sum(y_predict_test == y_test)*0.5/(y_test.shape[0]*1.0)

    y_predict_train = clf.predict(X_train)
    rbf_train = sum(y_predict_train == y_train)*0.5/(y_train.shape[0]*1.0)
    print('Average train one day score is ',rbf_train)
    print('Average test one day score is ',rbf_test)


