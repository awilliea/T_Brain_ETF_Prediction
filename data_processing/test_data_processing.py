
# coding: utf-8

# In[1]:


from data_processing import *

#get df data
#single_stock is 個股資料
#ETFtable is the integration data of all ETF
#features concludes ['orderbook','bond','TXF','EXF','FXF','EXF/FXF','Nikkei',
#                   'S&P 500','Russell 2000','Dow Jones','VIX','put call ratio','3種法人的未平倉量']

def test_get_dflist():
    
    single_stock,ETFtable,features = get_dflist()
    
    return single_stock,ETFtable,features


#Get Target ETF data
#code list = [50,51,52,53,54,55,56,57,58,59,6201,6203,6204,6208,690,692,701,713]
#e.g.  Yuanta50 = extract_target_ETF(ETFtable,50)

def test_extract_target_ETF(ETFtable,ETFcode):
    
    ETF_price = extract_target_ETF(ETFtable,ETFcode)
    
    return ETF_price


#Get the daily return dataframe of target ETF 
#backday means that how long the past data you want
#e.g. Yunta50_5 = get_change_rate(Yunta50,5) (conclude the past five data of daily return of Yunta50)

def test_get_change_rate(df,backday):
    
    df_backday = get_change_rate(df,backday)
    
    return  df_backday


#Get the dataframe of target ETF 
#backday means that how long the past data you want
#e.g. Yunta50_5(Yunta50,5) (conclude the past five data of Yunta50)

def test_getbackday(df,backday):
    
    df_backday = get_backday(df,backday)
    
    return  df_backday


#Get the rise,fall,noise data with interval of the 'close price' data of the df
#You should use the daily return data for the df variable
#Rise data is 1 by daily return > 0 + interval*std
#Fall data is 1 by daily return > 0 + interval*std
#e.g Yunta50_rf = get_rise_fall(Yunta50_rate,0.25)   

def test_get_rise_fall(df,interval):
    
    df_rf = get_rise_fall(df,interval)
    
    return df_rf


#Get the mean dataframe for window day of the df's columnname data
# 1 means that day is on the rising side
# -1 means that day is on the falling side
# Window decides the time we calulate for the mean data
# Days decides the daily return we chooses for the past mean data,if over halfs of the days data are positive,return 1 ,otherwise,-1
#e.g. Yunta50_mrf = get_mean_rf(Yunta50,'close',20,5)

def test_get_mean_rf(df,columnname,window,days):
    
    mrf = get_mean_rf(df,columnname,window,days)
    
    return mrf


#Get the df data into the category data,and concat the df_rf data
# df_rf is the rise,fall data due to the get_rise_fall function
# ca decides the number of category 
#e.g. Yunta50_ca = get_ca(Yunta50,df_rf,4)

def test_get_ca(df,df_rf,ca):
    
    df_ca = get_ca(df,df_rf,ca)
    
    return df_ca


#Get the average df data into the category data,and concat the df_rf data
# df_rf is the rise,fall data due to the get_rise_fall function
# ca decides the number of category 
# window decides the number of the past day using for get the mean data
#e.g. Yunta50_ca = get_ca(Yunta50,df_rf,10,4)

def test_get_av_ca(df,df_rf,window,ca):
    
    df_ca = get_ca(df,df_rf,window,ca)
    
    return df_ca

# Get all of the target data from ETFtable and ETF code
# ETFprice : the data of target ETF
# ETF_rate : the  daily return data of the target ETF
# mrf : the status of rising side(1) or falling side(1) decided by the 20day average line from the target ETF close price
# e.g. Yunta50, Yunta50_rate , Yunta50_mrf = get_targetdata(ETFtable,50)
def test_get_targetdata(ETFtable,code):
    
    ETFprice,ETF_rate,mrf = get_targetdata(ETFtable,code)
    
    return ETFprice,ETF_rate,mrf


# Get df_total(concat all dataframe of df_list and set extra null column for all features) 
# trf(Target ETF true rise fall data) rf(Target ETF interval rise and fall)
# df_list : all data you want to use for prediction(must put ETFprice at the df_list[0])
# df_r : boolean list ,True means you want the daily return of the data,False means the original data
# target : daily return of the Target data
# interval : float
# window : the number of the past day using for get the mean data
# ca : the number of category 
# day : the day you want to predict(1 means the next day)

def test_get_dataframe_null(df_list,df_r,target,interval,window,ca=4,day = 1):
    
    df_total ,trf , rf = get_dataframe_null(df_list,df_r,target,interval,window,ca=4,day = 1)
    
    return df_total ,trf , rf


# Almost the same as get_dataframe_null,but this time we drop all of the row which contains NAN

def test_get_dataframe_drop(df_list,df_r,target,interval,window,ca=4,day = 1):
    
    df_total ,trf , rf = get_dataframe_null(df_list,df_r,target,interval,window,ca=4,day = 1)
    
    return df_total ,trf , rf

# Almost the same as get_dataframe_drop,but this time we make every feature split to two features,one means on the rising side,
# the other is on the falling side

def test_get_dataframe_rf(df_list,df_r,target,interval,window,ca=4,day = 1):
    
    df_total ,trf , rf = get_dataframe_null(df_list,df_r,target,interval,window,ca=4,day = 1)
    
    return df_total ,trf , rf

