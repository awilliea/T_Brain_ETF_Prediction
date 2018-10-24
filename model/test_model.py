
# coding: utf-8

# In[4]:


from model import *

# rolling by pred_day
class Test_Model_Selection_Pre:
    
    #Initialize the class 
    # models : a dictionary decides which model you want to use
    # model_grid_params : a dictionary decides which parameters you want to optimize
    # stock_5 : train dataframe list which contains the dataframe we want to train of monday to friday
    # rf_5 : test dataframe list which contains the dataframe we want to test of monday to friday
    # latest_day : the size of training data
    # pred_day : the size of test data
    # day : same number of the length of stock_5

    def __init__(self,models,model_grid_params,stock_5,rf_5,latest_day,pred_day,day):
        
        pip = Model_Selection_Pre(models,model_grid_params,stock_5,rf_5,latest_day,pred_day,day)
        
        
    
    # Get the best parameters of every model and save them into self.grid[key] , save cv score into self.cv_acc[key]
    # key : the name of the model we use
    
    def test_Grid_fit(self,X_train,y_train,cv = 2,scoring = 'accuracy'):
        
        self.Grid_fit(X_train, y_train, cv = 2, scoring = 'accuracy')
         
    
    # Get the test accuracy score,fscore,predictions,test data 
    # and save them into self.acc[key],self.fscore[key],self.predict_values[key],self.true_values[key]
    
    def test_model_fit(self,X_train, y_train, X_test, y_test):
        self.model_fit(X_train, y_train,X_test,y_test)
        
            
    # Start running the model,and get the final score to the self.summary_day
    # Rolling role : check the length of data,and classify to the 6th cv , 2th cv or 1th cv
    # Rolling way : rolling by the pred_day
    
    def test_pipline(self):
        pip.pipline()
        
   
    # Get the predictions of the test data
    def test_model_pre(self,X_train,y_train,X_test):
        pip.model_pre(X_train,y_train,X_test)
            
    
    
    # Get the predictions of the test data from monday to friday
    def test_pipline_pre(self):
        pip.pipline_pre()
            
    
    # initialize the list
    def test_set_list(self):
        pip.set_list()
        
        
    # initialize the day list       
    def test_set_list_day(self):
        pip.set_list()
        
    
    # Get the score summary of the 'Estimator','Accuracy_mean','Accuracy_std','Accuracy_max','Accuracy_min','F_score'
    # and return the sorted dataframe
    def test_score_summary(self,sort_by):
        pip.score_summary(sort_by)
        return summary.sort_values(by = [sort_by], ascending=False)
    
    # Get the score summary when the cv time is only once
    def score_summary_1cv(self,sort_by):
        pip.score_summary_1cv(sort_by)
        return summary.sort_values(by = [sort_by], ascending=False)
          
    def print_(self):

        print (self.predict_values)

        
        
# Almost the same as Model_Selection_Pre but rolling by latest_day
# e.g Our data is [1,2,...,35] and pre_day is 5 ,latest_day is 10
# Model_Selection_Pre : 
#       1st training [1,..,10] test [11,..,15]
#       2nd training [15,..,25] test [26,..,30]
# Model_Selection_Las:
#       1st training [1,..,10] test [11,..,15]
#       2nd training [20,..,30] test [30,..,35]
class Test_Model_Selection_Las:

    def __init__(self,models,model_grid_params,stock_5,rf_5,latest_day,pred_day,day):
        
        pip = Model_Selection_Pre(models,model_grid_params,stock_5,rf_5,latest_day,pred_day,day)

    def test_Grid_fit(self,X_train,y_train,cv = 2,scoring = 'accuracy'):
        
        self.Grid_fit(X_train, y_train, cv = 2, scoring = 'accuracy')

    def test_model_fit(self,X_train, y_train, X_test, y_test):
        self.model_fit(X_train, y_train,X_test,y_test)
        
    def test_pipline(self):
        pip.pipline()
        
    def test_model_pre(self,X_train,y_train,X_test):
        pip.model_pre(X_train,y_train,X_test)

    def test_pipline_pre(self):
        pip.pipline_pre()

    def test_set_list(self):
        pip.set_list()

    def test_set_list_day(self):
        pip.set_list()
    
    def test_score_summary(self,sort_by):
        pip.score_summary(sort_by)
        return summary.sort_values(by = [sort_by], ascending=False)
    
    def score_summary_1cv(self,sort_by):
        pip.score_summary_1cv(sort_by)
        return summary.sort_values(by = [sort_by], ascending=False)
          
    def print_(self):

        print (self.predict_values)

