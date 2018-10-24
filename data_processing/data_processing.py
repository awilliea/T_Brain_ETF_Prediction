
# coding: utf-8

# In[1]:


# Load in our libraries
import pandas as pd
import numpy as np
import time
import datetime as dt
import math
import os

# In[2]:


def extract_target_index(etf_propotion_filepath,stock):
    ETF_propotion = pd.read_excel(etf_propotion_filepath)
    ETF_target = pd.DataFrame(columns = stock.columns)
    
    for code in ETF_propotion['code'].values:
        temp = stock[stock['code']== code]
        ETF_target = pd.concat([ETF_target,temp],axis = 0)
    
    #process data type
    ETF_target['date'] = ETF_target['date'].map(lambda x:dt.datetime.strptime(str(x),'%Y%m%d'))
    ETF_target['open'] = ETF_target['open'].map(lambda x:x if (type(x)==float) else float(x.replace(',','')) )
    ETF_target['high'] = ETF_target['high'].map(lambda x:x if (type(x)==float) else float(x.replace(',','')) )
    ETF_target['low'] = ETF_target['low'].map(lambda x:x if (type(x)==float) else float(x.replace(',','')) )
    ETF_target['close'] = ETF_target['close'].map(lambda x:x if (type(x)==float) else float(x.replace(',','')) )
    ETF_target['volume'] = ETF_target['volume'].map(lambda x:x if (type(x)==float) else float(x.replace(',','')) )
    
    #map with propotion (first column:date , last column:propotion , middle: features)
    ETF_target['propotion'] = ETF_target['code'].map(lambda x: ETF_propotion[ETF_propotion['code']==x]['propotion'].values[0])
    ETF_target.drop(['code','name'],axis = 1,inplace = True)
    
    for columnname in ETF_target.columns[1:-1]:
        ETF_target[columnname] = ETF_target[columnname] * ETF_target['propotion'] * 0.01
    ETF_target.drop(['propotion'],axis = 1,inplace = True)
    
    #for constituent stock into index
    ETF_target = ETF_target.groupby(['date']).sum()
    
    #rename columns
    ETF_target = ETF_target.rename(columns = {'open':'i_open','high':'i_high','low':'i_low','close':'i_close','volume':'i_volume'})
    
    return ETF_target 
    
def load_txt(filepath):
    file = []
    with open(filepath,'r') as filein:
        for line in filein:
            file.append(line.strip('\n').split(','))
    file = np.array(file)

    df = pd.DataFrame(file[1:],columns=['date','open','high','low','close','volume'])
    df['date'] = df['date'].map(lambda x:dt.datetime.strptime(str(x),'%Y/%m/%d'))
    df['open'] = df['open'].map(lambda x:x if (type(x)==float) else float(x.replace(',','')) )
    df['high'] = df['high'].map(lambda x:x if (type(x)==float) else float(x.replace(',','')) )
    df['low'] = df['low'].map(lambda x:x if (type(x)==float) else float(x.replace(',','')) )
    df['close'] = df['close'].map(lambda x:x if (type(x)==float) else float(x.replace(',','')) )
    df['volume'] = df['volume'].map(lambda x:x if (type(x)==float) else float(x.replace(',','')) )
    df = df.set_index('date')
    return df

def load_txt_TXF(filepath):
    file = []
    with open(filepath,'r') as filein:
        for line in filein:
            file.append(line.strip('\n').split(','))
    file = np.array(file)

    df = pd.DataFrame(file[1:],columns=['date','time','open','high','low','close','volume'])
    df.drop(['time'],axis = 1,inplace = True)
    df['date'] = df['date'].map(lambda x:dt.datetime.strptime(str(x),'%Y/%m/%d'))
    df['open'] = df['open'].map(lambda x:x if (type(x)==float) else float(x.replace(',','')) )
    df['high'] = df['high'].map(lambda x:x if (type(x)==float) else float(x.replace(',','')) )
    df['low'] = df['low'].map(lambda x:x if (type(x)==float) else float(x.replace(',','')) )
    df['close'] = df['close'].map(lambda x:x if (type(x)==float) else float(x.replace(',','')) )
    df['volume'] = df['volume'].map(lambda x:x if (type(x)==float) else float(x.replace(',','')) )
    df = df.set_index('date')
    return df

def load_txt_2(filepath):
    file = []
    with open(filepath,'r') as filein:
        for line in filein:
            file.append(line.strip('\n').split(','))
    file = np.array(file)

    df = pd.DataFrame(file[1:],columns=['date','time','open','high','low','close','volume'])
    df['date'] = df['date'].map(lambda x:dt.datetime.strptime(str(x),'%Y/%m/%d'))
    df['open'] = df['open'].map(lambda x:x if (type(x)==float) else float(x.replace(',','')) )
    df['high'] = df['high'].map(lambda x:x if (type(x)==float) else float(x.replace(',','')) )
    df['low'] = df['low'].map(lambda x:x if (type(x)==float) else float(x.replace(',','')) )
    df['close'] = df['close'].map(lambda x:x if (type(x)==float) else float(x.replace(',','')) )
    df['volume'] = df['volume'].map(lambda x:x if (type(x)==float) else float(x.replace(',','')) )
    df = df.set_index('date').drop(['time'],axis = 1)
    return df

def load_csv(filename):
    df = pd.read_csv(filename).drop(['Adj Close'],axis = 1)
    columnnames = df.columns
    df = df.rename(columns = {columnnames[0]:'date',columnnames[1]:'open',columnnames[2]:'high',columnnames[3]:'low',
                             columnnames[4]:'close',columnnames[5]:'volume'})
    df['date'] = df['date'].map(lambda x:dt.datetime.strptime(str(x),'%Y/%m/%d'))
    df['open'] = df['open'].map(lambda x:x if (type(x)==float or type(x)==int) else float(x.replace(',','')) )
    df['high'] = df['high'].map(lambda x:x if (type(x)==float or type(x)==int) else float(x.replace(',','')) )
    df['low'] = df['low'].map(lambda x:x if (type(x)==float or type(x)==int) else float(x.replace(',','')) )
    df['close'] = df['close'].map(lambda x:x if (type(x)==float or type(x)==int) else float(x.replace(',','')) )
    df['volume'] = df['volume'].map(lambda x:x if (type(x)==float or type(x)==int) else float(x.replace(',','')) )
    df = df.set_index('date')
    
    return df
	
def load_csv_3(filename):
    df = pd.read_excel(filename).iloc[1:,:]
    columnnames = df.columns
    df = df.rename(columns = {'11TX 自營商－臺指期':'DOI','12TX 投信－臺指期':'IOI','13TX 外資－臺指期':'FOI'})
    df['DOI'] = df['DOI'].map(lambda x:x if (type(x)==float or type(x)==int) else float(x.replace(',','')) )
    df['IOI'] = df['IOI'].map(lambda x:x if (type(x)==float or type(x)==int) else float(x.replace(',','')) )
    df['FOI'] = df['FOI'].map(lambda x:x if (type(x)==float or type(x)==int) else float(x.replace(',','')) )
    
    return df
	
def ETF_data_processing(filepath):
    ETFtable = pd.read_csv(filepath,encoding=' big5-hkscs ').rename(columns={'代碼':'code','日期':'date','中文簡稱':'name','開盤價(元)':'open','最高價(元)':'high','最低價(元)':'low','收盤價(元)':'close','成交張數(張)':'volume'})
   
    #processing data type 
    ETFtable['date'] = ETFtable['date'].map(lambda x:dt.datetime.strptime(str(x),'%Y%m%d'))
#     ETFtable['week'] = ETFtable['date'].map(lambda x:x.isoweekday())
    ETFtable['close'] = ETFtable['close'].map(lambda x:x if (type(x)==float) else float(x.replace(',','')) )
    ETFtable['open'] = ETFtable['open'].map(lambda x:x if (type(x)==float) else float(x.replace(',','')) )
    ETFtable['high'] = ETFtable['high'].map(lambda x:x if (type(x)==float) else float(x.replace(',','')) )
    ETFtable['low'] = ETFtable['low'].map(lambda x:x if (type(x)==float) else float(x.replace(',','')) )
    ETFtable['volume'] = ETFtable['volume'].map(lambda x:x if (type(x)==float) else float(x.replace(',','')) )
    
    return ETFtable
def extract_target_ETF(ETFtable,ETFcode):
    ETF_target = ETFtable[ETFtable['code']==ETFcode].drop(['code','name'],axis = 1).groupby(['date']).sum()
    
    return ETF_target

def get_dflist():
    
    data = pd.read_csv(os.path.join("dataset_all","tsharep.csv"),encoding=' big5-hkscs ').rename(columns={'代碼':'code','日期':'date','中文簡稱':'name','開盤價(元)':'open','最高價(元)':'high','最低價(元)':'low','收盤價(元)':'close','成交張數(張)':'volume'})
    # Taiwan50 = extract_target_index('ETF50.xlsx',data)
    TXF = load_txt_TXF(os.path.join("dataset_all",'TXF-1-300-分鐘.txt'))
    EXF = load_txt_2(os.path.join("dataset_all",'EXF-1-1-日.txt'))
    FXF = load_txt_2(os.path.join("dataset_all",'FXF-1-1-日.txt'))
    E_F = EXF/FXF
    Nikkei = load_csv(os.path.join("dataset_all",'Nikkei225.csv'))
    VIX = load_csv(os.path.join("dataset_all",'VIX.csv')).drop(['volume'],axis = 1)
    Russell = load_csv(os.path.join("dataset_all",'Russell2000.csv'))
    SP = load_csv(os.path.join("dataset_all",'S&P500.csv'))
    DJ = load_csv(os.path.join("dataset_all",'Dow Jones.csv'))
    pcr = load_txt(os.path.join("dataset_all",'put_call_ratio-日-成交價.txt')).drop(['volume','open','low'],axis = 1)
    OI = load_csv_3(os.path.join("dataset_all",'OI.xlsx'))
    orderbook_ = pd.read_hdf(os.path.join("dataset_all",'twse_orderbook_open.h5')).rename(columns = {'時間':'date',
                                                                        '累積委託買進筆數':'ask_vol',
                                                                        '累積委託買進數量':'ask_count',
                                                                        '累積委託賣出筆數':'bid_vol',
                                                                        '累積委託賣出數量':'bid_count',
                                                                        '累積成交筆數':'vol',
                                                                        '累積成交數量':'count',
                                                                        '累積成交金額':'amount'})
    orderbook = orderbook_.set_index('date')

    orderbook = orderbook.groupby('date').sum().drop(['vol','count','amount'],axis = 1)
    ETFtable = ETF_data_processing(os.path.join("dataset_all",'tetfp.csv'))



    fiveyear_bond = pd.read_excel(os.path.join("dataset_all",'five_year_bond.xls')).rename(columns = {'FRED Graph Observations':'date','Unnamed: 1':'five_year'}).set_index('date').iloc[10:]
    tenyear_bond = pd.read_excel(os.path.join("dataset_all",'ten_year_bond.xls')).rename(columns = {'FRED Graph Observations':'date','Unnamed: 1':'ten_year'}).set_index('date').iloc[10:]
    bond = pd.concat([fiveyear_bond,tenyear_bond],axis = 1).rename(columns = {'DGS5':'DGSfive' ,'DGS10':'DGSten'})
    return data,ETFtable,[orderbook,bond,TXF,EXF,FXF,E_F,Nikkei,SP,Russell,DJ,VIX,pcr,OI]

# In[10]:


def get_change_rate(stock,back_day = 1):
    amount_of_features = stock.columns.shape[0]
    for back in range(back_day+1):
        if(back == 0):
            stock_rate = (stock - stock.shift(back+1).fillna(0))/stock
            stock_temp = stock_rate.copy()
        else:
            stock_shift = stock_temp.shift(back).fillna(0)
            
            #rename columns
            column_names = list(stock_temp.columns)
            dict_ = {}
            for column_name in column_names:
                dict_[column_name] = column_name+f'_{back}'
                
            stock_shift = stock_shift.rename(columns = dict_)
            
            stock_rate = pd.concat([stock_rate,stock_shift],axis = 1)
    return stock_rate[back_day+1:]

def get_backday(stock,backday = 1):
    amount_of_features = stock.columns.shape[0]
    for back in range(backday+1):
        if (back == 0):
            stock_rate = stock.copy()
            stock_temp = stock.copy()
        else:
            stock_shift = stock_temp.shift(back)
            
            #rename columns
            column_names = list(stock_temp.columns)
            dict_ = {}
            for column_name in column_names:
                dict_[column_name] = column_name+f'_{back}'
                
            stock_shift = stock_shift.rename(columns = dict_)
            
            stock_rate = pd.concat([stock_rate,stock_shift],axis = 1)
    return stock_rate[backday+1:]

def isnoise(rate,describe,interval):
    if (rate <= 0+describe[2]*interval and rate >= 0-describe[2]*interval):
        return 1
    else :
        return 0
    
def isrise(rate,describe,interval):
    if (rate > 0+describe[2]*interval):
        return 1
    else :
        return 0
    
def isfall(rate,describe,interval):
    if(rate < 0-describe[2]*interval):
        return 1
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
    
    rise = []
    fall = []
    noise = []
    
    df_rf = pd.DataFrame(index=df.index)
    for day in range(df.shape[0]):
        rise.append(isrise(df['close'][day],df_rise_fall_d,interval))
        fall.append(isfall(df['close'][day],df_rise_fall_d,interval))
        noise.append(isnoise(df['close'][day],df_rise_fall_d,interval))

    df_rf['isrise'] = rise
    df_rf['isfall'] = fall
    df_rf['isnoise'] = noise
    return df_rf

def get_mean_rate_plot(mean_20day_diff,day,window,mean_20day):
    if(day-window < 20):
        return mean_20day_diff[0]
    data = mean_20day_diff[day - window : day]
    rise = sum(data > 0)
    rate = rise/(window*1.0)
    if (rate > 0.5 ):
        return mean_20day[day]+1
    else :
        return mean_20day[day] -1
    
def get_mean_rate(mean_20day_diff,day,window,mean_20day):
    if(day-window < 20):
        return mean_20day_diff[0]
    data = mean_20day_diff[day - window : day] # not include today
    rise = sum(data > 0)
    rate = rise/(window*1.0)
    if (rate > 0.5 ):
        return 1
    else :
        return -1
    
def get_mean_rf(stock,columnname,window,days):
    mean_20day = stock[columnname].rolling(window = window).mean()
    mean_20day_diff = ((mean_20day - mean_20day.shift(1) )/mean_20day.shift(1))*100.0
    mean_rise_fall = []
    for day in range(mean_20day_diff.shape[0]):
        mean_rise_fall.append(get_mean_rate(mean_20day_diff,day,days,mean_20day))
    return np.array(mean_rise_fall)

def category(stock_column,ca ):
    category = pd.qcut(stock_column.rank(method='first'),ca,labels = [_+1 for _ in range(ca)]).astype(np.float64)
    return category

def get_ca(stock,rf_df,ca ):
    stock_ca = stock.copy().astype(np.float64)
    for columnname in stock_ca.columns:
        stock_ca[columnname] = category(stock_ca[columnname] , ca)
        
    stock_ca = pd.concat([stock_ca,rf_df.loc[stock_ca.index[0]:stock_ca.index[-1]]],axis = 1).dropna(axis = 0,how = 'any')
    
    return stock_ca

# rfdata 表示isnrise isfall isnoise的來源
def get_av_ca(stock,rf_df,window,ca):
    amount_of_features = int(stock.columns.shape[0] / window)
    stock_av = stock.iloc[:,:amount_of_features].copy()
    for index,columnname in enumerate(stock_av.columns):
        for backday in range(window-1):
            stock_av[columnname] += stock[columnname.replace(columnname[-1],f'{backday+2}')]
        stock_av[columnname] /= amount_of_features
        stock_av[columnname] = category(stock_av[columnname],ca)
    
    stock_av = pd.concat([stock_av,rf_df.loc[stock_av.index[0]:stock_av.index[-1]]],axis = 1).dropna(axis = 0,how = 'any')
    
    return stock_av


def get_dataframe_null(df_list,df_r,target,interval,window,ca=4,day = 1):
    df_list_rate = []
    df_rf = get_rise_fall(target,interval).shift(-1*day+1)
    
    #concat the df in the df_list
    for index,df in enumerate(df_list):
        if(df_r[index]):
            df_rate = get_change_rate(df,window).iloc[:,df.shape[1]:]
        else:
            df_rate = get_backday(df,window).iloc[:,df.shape[1]:]
            
        df_av  = get_av_ca(stock=df_rate,rf_df=df_rf,ca=ca,window = window).iloc[:,:-3]
        df_colname = list(map(lambda x:  df_name[index]+'_'+x,df_av.columns))
        df_dum = pd.get_dummies(df_av,prefix=df_colname,columns=df_av.columns)
        df_list_rate.append(df_dum)
        
    df_all = pd.concat(df_list_rate,axis =1)
    df_all['mean_rise_fall'] = mrf
    
    #create null columns for the null data
    df_null = df_all.copy()
    for columnname in df_all.columns:
        df_null[columnname+'isnull'] = list(map(lambda x: int(x),df_all.isna()[columnname]))
    df_null.drop(df_all.columns,axis = 1,inplace = True)

    df_total = pd.concat([df_all,df_null],axis = 1).dropna(axis = 0,how = 'any')
    df_rf_ = df_rf.loc[df_total.index]
    #get target rise fall data
    Y = df_list[0].copy()
    Y['trf'] = np.sign((df_list[0]-df_list[0].shift(1))['close'].values)
    Y = Y.shift(-1*day+1)
    trf = Y.loc[df_total.index]['trf']

    return [df_total ,trf , df_rf_]

def get_dataframe_drop(df_list,df_r,target,interval,window,ca=4,day = 1):
    df_list_rate = []
    df_rf = get_rise_fall(target,interval).shift(-1*day+1)
    
    #concat the df in the df_list
    for index,df in enumerate(df_list):
        if(df_r[index]):
            df_rate = get_change_rate(df,window).iloc[:,df.shape[1]:]
        else:
            df_rate = get_backday(df,window).iloc[:,df.shape[1]:]
            
        df_av  = get_av_ca(stock=df_rate,rf_df=df_rf,ca=ca,window = window).iloc[:,:-3]
        df_colname = list(map(lambda x:  df_name[index]+'_'+x,df_av.columns))
        df_dum = pd.get_dummies(df_av,prefix=df_colname,columns=df_av.columns)
        df_list_rate.append(df_dum)
        
    df_all = pd.concat(df_list_rate,axis =1)
    df_all['mean_rise_fall'] = mrf
    
    df_total = df_all.dropna(axis = 0,how = 'any')
    df_rf_ = df_rf.loc[df_total.index]
    #get target rise fall data
    Y = df_list[0].copy()
    Y['trf'] = np.sign((df_list[0]-df_list[0].shift(1))['close'].values)
    Y = Y.shift(-1*day+1)
    trf = Y.loc[df_total.index]['trf']

    return [df_total,trf,df_rf_]

def get_dataframe_rf(df_list,df_r,target,interval,window,ca=4,day = 1):
    df_list_rate = []
    df_rf = get_rise_fall(target,interval).shift(-1*day+1)
    
    #concat the df in the df_list
    for index,df in enumerate(df_list):
        if(df_r[index]):
            df_rate = get_change_rate(df,window).iloc[:,df.shape[1]:]
        else:
            df_rate = get_backday(df,window).iloc[:,df.shape[1]:]
            
        df_av  = get_av_ca(stock=df_rate,rf_df=df_rf,ca=ca,window = window).iloc[:,:-3]
        df_colname = list(map(lambda x:  df_name[index]+'_'+x,df_av.columns))
        df_dum = pd.get_dummies(df_av,prefix=df_colname,columns=df_av.columns)
        df_list_rate.append(df_dum)
        
    df_all = pd.concat(df_list_rate,axis =1)
    df_all['mean_rise_fall'] = mrf
    
    df_total = df_all.dropna(axis = 0,how = 'any')
    
    #split rise and fall
    dicu = {}
    dicd = {}
    for index, col in enumerate(df_total.columns[:-1]):
        dicu[col] = col+'_r'
        dicd[col] = col+'_f'

    up = df_total[df_total['mean_rise_fall']==1].rename(columns = dicu).iloc[:,:-1]
    dn = df_total[df_total['mean_rise_fall']==-1].rename(columns = dicd).iloc[:,:-1]
    df_total_ud = pd.concat([up,dn],axis = 1).sort_index().fillna(0)
    df_rf_ = df_rf.loc[df_total.index]
	
    #get target rise fall data
    Y = df_list[0].copy()
    Y['trf'] = np.sign((df_list[0]-df_list[0].shift(1))['close'].values)
    Y = Y.shift(-1*day+1)
    trf = Y.loc[df_total_ud.index]['trf']

    return [df_total_ud,trf,df_rf_]
	

def get_targetdata(ETFtable,code):
    ETF_price = extract_target_ETF(ETFtable,code)
    ETF_rate = get_change_rate(ETF_price)
    mrf = pd.Series(get_mean_rf(ETF_price ,'close',20,5),index=ETF_price.index)
    return [ETF_price,ETF_rate,mrf]