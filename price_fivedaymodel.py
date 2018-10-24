
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

from price_onedaymodel import LinearRegressionReg,normalize_data,denormalize


#load data
def load_data_feature(stock, seq_len,feature,day=0):
    amount_of_features = len(stock.columns) 
    data = stock.as_matrix() 
    sequence_length = seq_len + 1 # index starting from 0
    result = []
    
    for index in range(len(data) - sequence_length): # maxmimum date = lastest date - sequence length
        result.append(data[index: index + sequence_length]) # index : index + 22days
    
    result = np.array(result)
    row = round(0.9 * result.shape[0]) # 90% split
    train = result[:int(row), :] # 90% date, all features 
    
    x_train = train[:, :-1] 
    y_train = train[:,-1,feature] 

    x_test = result[int(row):, :-1] 
    y_test = result[int(row):, -1,feature]

    
    x_train = x_train.reshape((x_train.shape[0],x_train.shape[1]*x_train.shape[2]))
    x_test =  x_test.reshape((x_test.shape[0],x_test.shape[1]*x_test.shape[2])) #shape(127,20,6) --> (127,120)
     

    return [x_train, y_train, x_test, y_test]


#transform the shape of features array into the shape of X data
def shape_transfrom(features):
    features_=[]
    for i in range(features.shape[1]):
        temp = np.array([])
        for feature in features:
            temp = np.hstack((temp,feature[i]))
        features_ .append(temp)
    
    features_ = np.array(features_)
    return features_


def make_features_model_first(stock,seq_len,lamb):
    amount_of_features = len(stock.columns)
    models = []
    errors = []
    features = []
    columns = stock.columns
    
    for i in range(amount_of_features): #load data and train
        x_train ,y_train ,x_test,y_test= load_data_feature(stock,seq_len,i)
        lr = LinearRegressionReg()
        lr.fit(x_train,y_train,lamb)
        models.append(lr)
        errors.append(lr.error(x_test,y_test))
        features.append(lr.predict(x_test))
#         print("Eout of %s is %f"%(columns[i],errors[i]))
        
    features_ = shape_transfrom(np.array(features))
    
    return [features_,models,x_test,y_test]

def predict_features(x_test,model,amount_of_features):
    errors = []
    features = []
    
    for i in range(amount_of_features): #load data and train
#         errors.append(lr.score(x_test,y_test))
        features.append(model[i].predict(x_test))
    features = np.array(features)
    features_ = shape_transfrom(features)
    
    return features_ 

def fiveday_predict(stock,seq_len,lamb):
    amount_of_features = len(stock.columns)
    features,model,x_test,y_test = make_features_model_first(stock,seq_len,lamb)
    y_predict = []
    y_predict.append(features[:,-1]) #shape = (127,1)
    length = x_test.shape[0]
    
    for day in range(4): 
        #add the prediction of features
        x_test_temp = []
        for i in range(length):
            x_test_temp.append(np.concatenate((x_test[i],features[i])))
        x_test = np.array(x_test_temp)[:,amount_of_features:] #still use the recent 20 days data

        #Predict features
        features_1 = predict_features(x_test,model,amount_of_features)
        y_predict.append(features_1[:,-1]) #take the close price
    return [np.array(y_predict),y_test]
    
def week_score(original_stock,stock,window,lamb):
    y_predict ,y_test= fiveday_predict(stock,window,lamb)
    
    length = y_test.shape[0]
    y_test = denormalize(original_stock,y_test,'close').reshape((length))
    week_score_ = []
    
    for day in range(5):
        y_test_temp = y_test[day:]
        y_predict_temp = denormalize(original_stock,y_predict[day],'close').reshape((length))[:length-day]
        temp = np.array(list(map(lambda x,y:((y-abs(x-y))/y)*0.5,y_predict_temp,y_test_temp)))
        week_score_.append(temp[:length-4])
    week_score_ = np.array(week_score_)
    
    week_score = []
    for week in range(week_score_.shape[1]):
        score = 0
        for day in range(week_score_.shape[0]):
            if (day == 0):
                score += week_score_[day][week]*0.1
            elif (day == 1):
                score += week_score_[day][week]*0.15
            elif (day == 2):
                score += week_score_[day][week]*0.2
            elif (day == 3):
                score += week_score_[day][week]*0.25
            elif (day == 4):
                score += week_score_[day][week]*0.3
        week_score.append(score)
    
    return [week_score,sum(week_score)/(len(week_score)*1.0)]




def five_day_prediction_18(ETFtable,ETF_list,window,lamb):
    ETF_score = []
    ETF_week_score = []
    for ETF_code in ETF_list:
        ETF_target = load_ETF(ETFtable,ETF_code)
        ETF_target_N = normalize_data(ETF_target)
        
        fiveday_score_array,fiveday_score = week_score(ETF_target,ETF_target_N,window,lamb)
        print(f'{ETF_code} average Fiveday_score is ',fiveday_score)
        ETF_score.append(fiveday_score)
        ETF_week_score.append(fiveday_score_array)
        
    print('Total ETF five day price score is',sum(ETF_score))    
    print('Average ETF five day price score is',sum(ETF_score)/18.0)
    return ETF_week_score

def five_day_prediction(ETF,window,lamb):
    ETF_target_N = normalize_data(ETF)
    fiveday_score_array,fiveday_score = week_score(ETF,ETF_target_N,window,lamb)
    
    print('Average ETF five day price score is',fiveday_score)
    return fiveday_score_array
	
def show_week_score_18(ETF_week_score):
    min_period = len(ETF_week_score[-1])
    regression_score = []
    
    for i in range(len(ETF_week_score)):
        regression_score.append(ETF_week_score[i][-1*min_period:])
    regression_score = np.array(regression_score)

    week_regression_score = []

    for week in range(regression_score.shape[1]):
        temp = 0
        for code in range(regression_score.shape[0]):
            temp += regression_score[code][week]
        week_regression_score.append(temp)
        
    length = len(week_regression_score)
    plt.title('Week score')
    plt.xlabel('week')
    plt.ylabel('score')
    plt.plot([_ for _ in range(length)],week_regression_score,label = 'score')
    plt.plot([_ for _ in range(length)],[9 for _ in range(length)],label = 'full_score')
    plt.legend()
    plt.show()
    
    return week_regression_score

def show_week_score(score):
    length = len(score)
    plt.title('Week score')
    plt.xlabel('week')
    plt.ylabel('score')
    plt.plot([_ for _ in range(length)],score,label = 'score')
    plt.plot([_ for _ in range(length)],[0.5 for _ in range(length)],label = 'full_score')
    plt.legend()
    plt.show()



