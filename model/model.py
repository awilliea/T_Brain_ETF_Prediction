
# coding: utf-8

# Load in our libraries
import pandas as pd
import numpy as np

import time
import datetime as dt
import warnings
warnings.filterwarnings('ignore')
from sklearn import preprocessing
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier,\
                              GradientBoostingClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

import math
import statistics


# rolling by pred_day
class Model_Selection_Pre:
    
    def __init__(self,models,model_grid_params,stock_5,rf_5,latest_day,pred_day,day):
        
        self.models = models
        self.model_grid = model_grid_params
        self.stock_5 = stock_5
        self.rf_5 = rf_5
        self.latest_day= latest_day
        self.pred_day = pred_day
        self.day = day
        self.keys = models.keys()
        self.best_score = {}
        self.grid = {}
        
        self.predict_values = {}
        self.cv_acc = {}
        self.acc = {}
        self.fscore = {}
        self.true_values = {}
        
        self.predict_values_day = {}
        self.cv_acc_day = {}
        self.acc_day = {}
        self.fscore_day = {}
        self.true_values_day = {}
        self.summary_day = []
        
    def Grid_fit(self,X_train,y_train,cv = 5,scoring = 'accuracy'):
        
        for key in self.keys:
            print ("Running GridSearchCV for %s" %(key))
            model = self.models[key]
            model_grid = self.model_grid[key]
            Grid = GridSearchCV(model, model_grid, cv = cv, scoring = scoring)
            Grid.fit(X_train,y_train) 
            self.grid[key] = Grid
            print (Grid.best_params_)
            print ('CV Best Score = %s'%(Grid.best_score_))
            self.cv_acc[key].append(Grid.best_score_)  
    
    def model_fit(self,X_train, y_train, X_test, y_test):
        
        for key in self.keys:
            print ("Running training & testing for %s." %(key))
            model = self.models[key]
            model.set_params(**self.grid[key].best_params_)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            #print 'Prediction latest 15 second = %s'%(predictions)
            self.predict_values[key].append(predictions.tolist())
            self.true_values[key].append(y_test.tolist())
            acc = metrics.accuracy_score(y_test,predictions)
            f_score = metrics.f1_score(y_test,predictions)
            print ('Accuracy = %s'%(acc))
            self.acc[key].append(acc)
            self.fscore[key].append(f_score)
            

    def pipline(self):
        
        self.set_list_day() # store day values
        for day in range(0,self.day,1):
            self.set_list() # store values
            print ('Day = %s'%(day+1))
            check1 = self.stock_5[-1].shape[0]-self.latest_day-self.pred_day*6
            check2 = self.stock_5[-1].shape[0]-self.latest_day-self.pred_day*2
            if (check1 > 0):
                for k in range(0,self.pred_day*5,self.pred_day):#9000-self.latest_day-600,self.pred_day):
                    i = k + self.stock_5[day].shape[0]-self.latest_day-self.pred_day*5
                    print ('--------------------Rolling Window Time = %s--------------------'%(k/self.pred_day))
                    # Train data
                    X_train = self.stock_5[day][i:i+self.latest_day]
                    data_train = self.rf_5[day][i:i+self.latest_day]

                    train_rise = data_train[:,-3]
                    train_fall = data_train[:,-2]
                    train_noise = data_train[:,-1]
                    y_train = train_rise

                    # Test data
                    X_test = self.stock_5[day][i + self.latest_day:i + self.latest_day + self.pred_day]
                    data_test = self.rf_5[day][i + self.latest_day:i + self.latest_day + self.pred_day]
                    test_rise = data_test[:,-3]
                    test_fall = data_test[:,-2]
                    test_noise = data_test[:,-1]
                    y_test = test_rise

                    #start = time.time()
                    self.Grid_fit(X_train, y_train, cv = 2, scoring = 'accuracy')
                    self.model_fit(X_train, y_train,X_test,y_test)
                    #end = time.time()
                    #print 'Total Time = %s'%(end - start)
            elif(check2 > 0):
                for k in range(0,self.pred_day*2,self.pred_day):
                    i = k + self.stock_5[day].shape[0]-self.latest_day-self.pred_day*2
                    print ('--------------------Rolling Window Time = %s--------------------'%(k/self.pred_day))
                    # Train data
                    X_train = self.stock_5[day][i:i+self.latest_day]
                    data_train = self.rf_5[day][i:i+self.latest_day]

                    train_rise = data_train[:,-3]
                    train_fall = data_train[:,-2]
                    train_noise = data_train[:,-1]
                    y_train = train_rise

                    # Test data
                    X_test = self.stock_5[day][i + self.latest_day:i + self.latest_day + self.pred_day]
                    data_test = self.rf_5[day][i + self.latest_day:i + self.latest_day + self.pred_day]
                    test_rise = data_test[:,-3]
                    test_fall = data_test[:,-2]
                    test_noise = data_test[:,-1]
                    y_test = test_rise

                    #start = time.time()
                    self.Grid_fit(X_train, y_train, cv = 2, scoring = 'accuracy')
                    self.model_fit(X_train, y_train,X_test,y_test)
                    #end = time.time()
                    #print 'Total Time = %s'%(end - start)
            else:
                
                i = 0
                print ('--------------------Rolling Window Time = %s--------------------'%(i/self.pred_day))
                # Train data
                X_train = self.stock_5[day][i:-1*self.pred_day]
                data_train = self.rf_5[day][i:-1*self.pred_day]

                train_rise = data_train[:,-3]
                train_fall = data_train[:,-2]
                train_noise = data_train[:,-1]
                y_train = train_rise

                # Test data
                X_test = self.stock_5[day][-1*self.pred_day:]
                data_test = self.rf_5[day][-1*self.pred_day:]
                test_rise = data_test[:,-3]
                test_fall = data_test[:,-2]
                test_noise = data_test[:,-1]
                y_test = test_rise

                #start = time.time()
                self.Grid_fit(X_train, y_train, cv = 2, scoring = 'accuracy')
                self.model_fit(X_train, y_train,X_test,y_test)
                #end = time.time()
                #print 'Total Time = %s'%(end - start)
                
            for key in self.keys:
                
                self.cv_acc_day[key].append(self.cv_acc[key])
                self.acc_day[key].append(self.acc[key])
                self.fscore_day[key].append(self.fscore[key])
                self.true_values_day[key].append(self.true_values[key])
                self.predict_values_day[key].append(self.predict_values[key])
            if (check2 > 0):
                self.summary_day.append(self.score_summary(sort_by = 'Accuracy_mean'))
            else:
                self.summary_day.append(self.score_summary_1cv(sort_by = 'Accuracy_mean'))
            
    def model_pre(self,X_train,y_train,X_test):
        for key in self.keys:
            print ("Running training & testing for %s." %(key))
            model = self.models[key]
            model.set_params(**self.grid[key].best_params_)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            #print 'Prediction latest 15 second = %s'%(predictions)
            self.predict_values[key].append(predictions.tolist())
            
        
    def pipline_pre(self):
        self.set_list_day() # store day values
        for day in range(0,self.day,1):
            self.set_list() # store values
            print ('Day = %s'%(day+1))
            
            # Train data
            X_train = self.stock_5[day]
            data_train = self.rf_5[day]
            
            train_rise = data_train[:,-3]
            train_fall = data_train[:,-2]
            train_noise = data_train[:,-1]
            y_train = train_rise
            
            #test
            X_test = self.stock_5[day][-1:,:]
            
            #start = time.time()
            self.Grid_fit(X_train, y_train, cv = 2, scoring = 'accuracy')
            self.model_pre(X_train, y_train , X_test)
            #end = time.time()
            #print 'Total Time = %s'%(end - start)
                
            for key in self.keys:
                
                self.cv_acc_day[key].append(self.cv_acc[key])
                self.acc_day[key].append(self.acc[key])
                self.fscore_day[key].append(self.fscore[key])
                self.true_values_day[key].append(self.true_values[key])
                self.predict_values_day[key].append(self.predict_values[key])
            
#             self.summary_day.append(self.score_summary(sort_by = 'Accuracy_mean'))
            
    
    
    def set_list(self):
        
        for key in self.keys:
            self.predict_values[key] = []
            self.cv_acc[key] = []
            self.acc[key] = []
            self.fscore[key] = []
            self.true_values[key] = []
            
    def set_list_day(self):
        
        for key in self.keys:
            self.predict_values_day[key] = []
            self.cv_acc_day[key] = []
            self.acc_day[key] = []
            self.fscore_day[key] = []
            self.true_values_day[key] = []
            
    def score_summary(self,sort_by):
        
        summary = pd.concat([pd.Series(list(self.acc.keys())),pd.Series(map(lambda x: sum(self.acc[x])/len(self.acc[x]), self.acc)),                             pd.Series(list(map(lambda x: statistics.stdev(self.acc[x]), self.acc))),                             pd.Series(list(map(lambda x: max(self.acc[x]), self.acc))),                             pd.Series(list(map(lambda x: min(self.acc[x]), self.acc))),                             pd.Series(list(map(lambda x: sum(self.fscore[x])/len(self.fscore[x]), self.fscore)))],axis=1)
        summary.columns = ['Estimator','Accuracy_mean','Accuracy_std','Accuracy_max','Accuracy_min','F_score']
        summary.index.rename('Ranking', inplace=True)
        return summary.sort_values(by = [sort_by], ascending=False)
    
    def score_summary_1cv(self,sort_by):
        
        summary = pd.concat([pd.Series(list(self.acc.keys())),pd.Series(map(lambda x: sum(self.acc[x])/len(self.acc[x]), self.acc)),                             
                             pd.Series(list(map(lambda x: max(self.acc[x]), self.acc))),\
                             pd.Series(list(map(lambda x: min(self.acc[x]), self.acc))),\
                             pd.Series(list(map(lambda x: sum(self.fscore[x])/len(self.fscore[x]), self.fscore)))],axis=1)
        summary.columns = ['Estimator','Accuracy_mean','Accuracy_max','Accuracy_min','F_score']
        summary.index.rename('Ranking', inplace=True)
        return summary.sort_values(by = [sort_by], ascending=False)
          
    def print_(self):

        print (self.predict_values)

        
        
# rolling by latest_day
class Model_Selection_Las:
    
    def __init__(self,models,model_grid_params,stock_5,rf_5,latest_day,pred_day,day,window = 1):
        
        self.week_score = []
        self.models = models
        self.model_grid = model_grid_params
        self.stock_5 = stock_5
        self.rf_5 = rf_5
        self.latest_day= latest_day
        self.pred_day = pred_day
        self.day = day
        self.keys = models.keys()
        self.best_score = {}
        self.grid = {}
        
        self.predict_values = {}
#         self.predict_values = {}
        self.cv_acc = {}
        self.acc = {}
        self.fscore = {}
        self.true_values = {}
        
        self.predict_values_day = {}
        self.cv_acc_day = {}
        self.acc_day = {}
        self.fscore_day = {}
        self.true_values_day = {}
        self.summary_day = []
        
    def Grid_fit(self,X_train,y_train,cv = 5,scoring = 'accuracy'):
        
        for key in self.keys:
            print ("Running GridSearchCV for %s" %(key))
            model = self.models[key]
            model_grid = self.model_grid[key]
            Grid = GridSearchCV(model, model_grid, cv = cv, scoring = scoring)
            Grid.fit(X_train,y_train) 
            self.grid[key] = Grid
            print (Grid.best_params_)
            print ('CV Best Score = %s'%(Grid.best_score_))
            self.cv_acc[key].append(Grid.best_score_)  
    
    def model_fit(self,X_train, y_train, X_test, y_test):
        
        for key in self.keys:
            print ("Running training & testing for %s." %(key))
            model = self.models[key]
            model.set_params(**self.grid[key].best_params_)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            #print 'Prediction latest 15 second = %s'%(predictions)
            self.predict_values[key].append(predictions.tolist())
            self.true_values[key].append(y_test.tolist())
            acc = metrics.accuracy_score(y_test,predictions)
            f_score = metrics.f1_score(y_test,predictions)
            print ('Accuracy = %s'%(acc))
            self.acc[key].append(acc)
            self.fscore[key].append(f_score)

    def pipline(self):
        
        self.set_list_day() # store day values
        for day in range(0,self.day,1):
            self.set_list() # store values
            print ('Day = %s'%(day+1))
            check1 = self.stock_5[-1].shape[0]-1000-self.pred_day-self.latest_day[day]
            check2 = self.stock_5[-1].shape[0]-100-self.pred_day-self.latest_day[day]-4
            if (check1 > 0):
                for k in range(0,1000,self.latest_day[day]):
                    i = k + self.stock_5[day].shape[0]-1000-self.pred_day-self.latest_day[day]
                    print ('--------------------Rolling Window Time = %s--------------------'%(k/self.latest_day[day]))
                    # Train data
                    X_train = self.stock_5[day][i:i+self.latest_day[day]]
                    data_train = self.rf_5[day][i:i+self.latest_day[day]]

                    train_rise = data_train[:,-3]
                    train_fall = data_train[:,-2]
                    train_noise = data_train[:,-1]
                    y_train = train_rise

                    # Test data
                    print(i)
                    X_test = self.stock_5[day][i + self.latest_day[day]:i + self.latest_day[day] + self.pred_day]
                    data_test = self.rf_5[day][i + self.latest_day[day]:i + self.latest_day[day] + self.pred_day]
                    test_rise = data_test[:,-3]
                    test_fall = data_test[:,-2]
                    test_noise = data_test[:,-1]
                    y_test = test_rise

                    #start = time.time()
                    self.Grid_fit(X_train, y_train, cv = 2, scoring = 'accuracy')
                    self.model_fit(X_train, y_train,X_test,y_test)
                    #end = time.time()
                    #print 'Total Time = %s'%(end - start)
            elif(check2 > 0):
                for k in range(0,100,self.latest_day[day]):
                    i = k + self.stock_5[day].shape[0]-100-self.pred_day-self.latest_day[day]
                    print ('--------------------Rolling Window Time = %s--------------------'%(k/latest_day[day]))
                    # Train data
                    X_train = self.stock_5[day][i:i+self.latest_day[day]]
                    data_train = self.rf_5[day][i:i+self.latest_day[day]]

                    train_rise = data_train[:,-3]
                    train_fall = data_train[:,-2]
                    train_noise = data_train[:,-1]
                    y_train = train_rise

                    # Test data
                    X_test = self.stock_5[day][i + self.latest_day[day]:i + self.latest_day[day] + self.pred_day]
                    data_test = self.rf_5[day][i + self.latest_day[day]:i + self.latest_day[day] + self.pred_day]
                    test_rise = data_test[:,-3]
                    test_fall = data_test[:,-2]
                    test_noise = data_test[:,-1]
                    y_test = test_rise

                    #start = time.time()
                    self.Grid_fit(X_train, y_train, cv = 2, scoring = 'accuracy')
                    self.model_fit(X_train, y_train,X_test,y_test)
                    #end = time.time()
                    #print 'Total Time = %s'%(end - start)
            else:
                
                i = 4
                print ('--------------------Rolling Window Time = %s--------------------'%(i/pred_day))
                # Train data
                X_train = self.stock_5[day][i:-1*self.pred_day]
                data_train = self.rf_5[day][i:-1*self.pred_day]

                train_rise = data_train[:,-3]
                train_fall = data_train[:,-2]
                train_noise = data_train[:,-1]
                y_train = train_rise

                # Test data
                X_test = self.stock_5[day][-1*self.pred_day:]
                data_test = self.rf_5[day][-1*self.pred_day:]
                test_rise = data_test[:,-3]
                test_fall = data_test[:,-2]
                test_noise = data_test[:,-1]
                y_test = test_rise

                #start = time.time()
                self.Grid_fit(X_train, y_train, cv = 2, scoring = 'accuracy')
                self.model_fit(X_train, y_train,X_test,y_test)
                #end = time.time()
                #print 'Total Time = %s'%(end - start)
                
            for key in self.keys:
                
                self.cv_acc_day[key].append(self.cv_acc[key])
                self.acc_day[key].append(self.acc[key])
                self.fscore_day[key].append(self.fscore[key])
                self.true_values_day[key].append(self.true_values[key])
                self.predict_values_day[key].append(self.predict_values[key])
            if (check2 > 0):
                self.summary_day.append(self.score_summary(sort_by = 'Accuracy_mean'))
            else:
                self.summary_day.append(self.score_summary_1cv(sort_by = 'Accuracy_mean'))
            
    def model_pre(self,X_train,y_train,X_test):
        for key in self.keys:
            print ("Running training & testing for %s." %(key))
            model = self.models[key]
            model.set_params(**self.grid[key].best_params_)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            #print 'Prediction latest 15 second = %s'%(predictions)
            self.predict_values[key].append(predictions.tolist())
            
        
    def pipline_pre(self):
        self.set_list_day() # store day values
        for day in range(0,self.day,1):
            self.set_list() # store values
            print ('Day = %s'%(day+1))
            
            # Train data
            X_train = self.stock_5[day][-1*self.latest_day[day]:]
            data_train = self.rf_5[day][-1*self.latest_day[day]:]
            
            train_rise = data_train[:,-3]
            train_fall = data_train[:,-2]
            train_noise = data_train[:,-1]
            y_train = train_rise
            
            #test 有BUG
            X_test = self.stock_5[day][-1*window:,:]
#             X_test = self.true_stock[-1:,:]
            
            #start = time.time()
            self.Grid_fit(X_train, y_train, cv = 2, scoring = 'accuracy')
            self.model_pre(X_train, y_train , X_test)
            #end = time.time()
            #print 'Total Time = %s'%(end - start)
                
            for key in self.keys:
                
                self.cv_acc_day[key].append(self.cv_acc[key])
                self.acc_day[key].append(self.acc[key])
                self.fscore_day[key].append(self.fscore[key])
                self.true_values_day[key].append(self.true_values[key])
                self.predict_values_day[key].append(self.predict_values[key])
            
#             self.summary_day.append(self.score_summary(sort_by = 'Accuracy_mean'))
            
    
    def pipline_pre_4(self):
        pro = [0.1,0.15,0.2,0.25,0.3]
        
        for week in range(4):
            self.set_list_day() # store day values
            score_day  = 0
            for day in range(0,self.day,1):
                self.set_list() # store values
                print ('Day = %s'%(day+1))
                i = -20+5*week
                print ('--------------------Rolling Window Time = %s--------------------'%(week))
                # Train data
                X_train = self.stock_5[day][i-self.latest_day[day]:i]
                data_train = self.rf_5[day][i-self.latest_day[day]:i]

                train_rise = data_train[:,-3]
                train_fall = data_train[:,-2]
                train_noise = data_train[:,-1]
                y_train = train_rise

                # Test data
                if(week == 3 and day == 4):
                    X_test = self.stock_5[day][i+day:]
                    data_test = self.rf_5[day][i+day:]
                    test_rise = data_test[:,-3]
                    test_fall = data_test[:,-2]
                    test_noise = data_test[:,-1]
                    y_test = test_rise
                else:
                    X_test = self.stock_5[day][i+day:i+day+1]
                    data_test = self.rf_5[day][i+day:i+day+1]
                    test_rise = data_test[:,-3]
                    test_fall = data_test[:,-2]
                    test_noise = data_test[:,-1]
                    y_test = test_rise
                    
                #start = time.time()
                self.Grid_fit(X_train, y_train, cv = 2, scoring = 'accuracy')
                self.model_fit(X_train, y_train,X_test,y_test)
                #end = time.time()
                #print 'Total Time = %s'%(end - start)
                
                score_day += (self.acc['RandomForestClassifier'][0]*pro[day]*0.5)
                
                for key in self.keys:
                    self.cv_acc_day[key].append(self.cv_acc[key])
                    self.acc_day[key].append(self.acc[key])
                    self.fscore_day[key].append(self.fscore[key])
                    self.true_values_day[key].append(self.true_values[key])
                    self.predict_values_day[key].append(self.predict_values[key])
                
            self.week_score.append(score_day)
            
            
    def set_list(self):
        
        for key in self.keys:
            self.predict_values[key] = []
            self.cv_acc[key] = []
            self.acc[key] = []
            self.fscore[key] = []
            self.true_values[key] = []
            
    def set_list_day(self):
        
        for key in self.keys:
            self.predict_values_day[key] = []
            self.cv_acc_day[key] = []
            self.acc_day[key] = []
            self.fscore_day[key] = []
            self.true_values_day[key] = []
            
    def score_summary(self,sort_by):
        
        summary = pd.concat([pd.Series(list(self.acc.keys())),pd.Series(map(lambda x: sum(self.acc[x])/len(self.acc[x]), self.acc)),\
                             pd.Series(list(map(lambda x: statistics.stdev(self.acc[x]), self.acc))),\
                             pd.Series(list(map(lambda x: max(self.acc[x]), self.acc))),\
                             pd.Series(list(map(lambda x: min(self.acc[x]), self.acc))),\
                             pd.Series(list(map(lambda x: sum(self.fscore[x])/len(self.fscore[x]), self.fscore)))],axis=1)
        summary.columns = ['Estimator','Accuracy_mean','Accuracy_std','Accuracy_max','Accuracy_min','F_score']
        summary.index.rename('Ranking', inplace=True)
        return summary.sort_values(by = [sort_by], ascending=False)
    
    def score_summary_1cv(self,sort_by):
        
        summary = pd.concat([pd.Series(list(self.acc.keys())),pd.Series(map(lambda x: sum(self.acc[x])/len(self.acc[x]), self.acc)),\
                             
                             pd.Series(list(map(lambda x: max(self.acc[x]), self.acc))),\
                             pd.Series(list(map(lambda x: min(self.acc[x]), self.acc))),\
                             pd.Series(list(map(lambda x: sum(self.fscore[x])/len(self.fscore[x]), self.fscore)))],axis=1)
        summary.columns = ['Estimator','Accuracy_mean','Accuracy_max','Accuracy_min','F_score']
        summary.index.rename('Ranking', inplace=True)
        return summary.sort_values(by = [sort_by], ascending=False)
          
    def print_(self):

        print (self.predict_values)