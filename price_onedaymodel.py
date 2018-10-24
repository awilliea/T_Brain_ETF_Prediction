
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
from load import load_ETF
import matplotlib.pyplot as plt

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

from sklearn import preprocessing


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


def one_day_prediction_18(ETFtable,ETF_list,window,lamb):
    ETF_score = []
    ETF_model = []
    for ETF_code in ETF_list:
        ETF_target = load_ETF(ETFtable,ETF_code)
        ETF_target_N = normalize_data(ETF_target)
        X_train, y_train, X_test, y_test = load_data(ETF_target_N, window)

        ### Model
        lr = LinearRegressionReg()
        lr.fit(X_train, y_train, lamb)

        ### score
        score = lr.score(X_test,y_test,ETF_target)
        print(f'{ETF_code} average one day score is',score)
        ETF_score.append(score)
        ETF_model.append(lr)
    ETF_features = ETF_target.columns    
    print('Total ETF one day price score is',sum(ETF_score))    
    print('Average ETF one day price score is',sum(ETF_score)/18.0)
    return [ETF_score , ETF_model,ETF_features]

def one_day_prediction(ETF,window,lamb):
    ETF_features = ETF.columns
    ETF_target_N = normalize_data(ETF)
    X_train, y_train, X_test, y_test = load_data(ETF_target_N, window)

    ### Model
    lr = LinearRegressionReg()
    lr.fit(X_train, y_train, lamb)

    ### score
    score = lr.score(X_test,y_test,ETF)
    print('Average one day score is',score)
    return [score , lr , ETF_features]
    

# Picture
def feature_importance(lr):
    plt.figure(figsize=(10,8))
    plt.title('Feature Importance')
    plt.xlabel('Feature')
    plt.ylabel('Weight')
    plt.bar([_ for _ in range(len(lr.get_w()))],lr.get_w())
    plt.show()
    
def feature_importance_scatter(lr,window,ETF_features):
    
    picture = [[_] for _ in range(window)]
    picture_label = []
    for i in picture:
        picture_label += i*10
    
    xlabel = list(ETF_features)

    # Scatter plot 
    trace = go.Scatter(
        y = lr.get_w(),
        x = xlabel*20,
        mode='markers',
        name = 'Importance',
        marker=dict(
            sizemode = 'diameter',
            sizeref = 1,
            size = 25,
    #       size= feature_dataframe['AdaBoost feature importances'].values,
            #color = np.random.randn(500), #set color equal to a variable
            color = lr.get_w(),
            colorscale='Portland',
            showscale=True
        ),
        text = xlabel*20
    )
    data = [trace]

    layout= go.Layout(
        autosize= True,
        title= 'Linear Regression Feature Importance on weight',
        hovermode= 'closest',
        xaxis= dict(
            title= 'Feature name',
    #         ticklen= 5,
    #         zeroline= False,
    #         gridwidth= 2,
        ),
        yaxis=dict(
            title= 'Feature Importance',
            ticklen= 5,
            gridwidth= 2
        ),
        showlegend= False
    )
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig,filename='scatter1')

    # Scatter plot 
    trace = go.Scatter(
        y = lr.get_w(),
        x = xlabel*20,
        mode='markers',
        name = 'Day',
        marker=dict(
            sizemode = 'diameter',
            sizeref = 1,
            size = 25,
    #       size= feature_dataframe['AdaBoost feature importances'].values,
            #color = np.random.randn(500), #set color equal to a variable
            color = picture_label,
            colorscale='Portland',
            showscale=True
        ),
        text = xlabel*20
    )
    data = [trace]

    layout= go.Layout(
        autosize= True,
        title= 'Linear Regression Feature Importance on day',
        hovermode= 'closest',
        xaxis= dict(
            title= 'Feature name',
    #         ticklen= 5,
    #         zeroline= False,
    #         gridwidth= 2,
        ),
        yaxis=dict(
            title= 'Feature Importance',
            ticklen= 5,
            gridwidth= 2
        ),
        showlegend= False
    )
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig,filename='scatter2')

