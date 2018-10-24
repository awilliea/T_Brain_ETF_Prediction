
# coding: utf-8

# In[9]:


import pandas as pd
import datetime as dt
import time
 
def load_stock(fname):
    print('load',fname)
    stock = pd.read_csv(fname,encoding=' big5-hkscs ').rename(columns={'代碼':'code','日期':'date','中文簡稱':'name','開盤價(元)':'open','最高價(元)':'high','最低價(元)':'low','收盤價(元)':'close','成交張數(張)':'volume'})
    
    return stock


def load_ETFtable(fname):
    print('load',fname)
    ETFtable = pd.read_csv(fname,encoding=' big5-hkscs ').rename(columns={'代碼':'code','日期':'date','中文簡稱':'name','開盤價(元)':'open','最高價(元)':'high','最低價(元)':'low','收盤價(元)':'close','成交張數(張)':'volume'})
   
    #processing data type 
    ETFtable['date'] = ETFtable['date'].map(lambda x:dt.datetime.strptime(str(x),'%Y%m%d'))
#     ETFtable['week'] = ETFtable['date'].map(lambda x:x.isoweekday())
    ETFtable['close'] = ETFtable['close'].map(lambda x:x if (type(x)==float) else float(x.replace(',','')) )
    ETFtable['open'] = ETFtable['open'].map(lambda x:x if (type(x)==float) else float(x.replace(',','')) )
    ETFtable['high'] = ETFtable['high'].map(lambda x:x if (type(x)==float) else float(x.replace(',','')) )
    ETFtable['low'] = ETFtable['low'].map(lambda x:x if (type(x)==float) else float(x.replace(',','')) )
    ETFtable['volume'] = ETFtable['volume'].map(lambda x:x if (type(x)==float) else float(x.replace(',','')) )
    
    return ETFtable


def load_ETF(ETFtable,ETFcode):
    print('load ETF',ETFcode)
    ETF_target = ETFtable[ETFtable['code']==ETFcode].drop(['code','name'],axis = 1).groupby(['date']).sum()
#     ETF_target = pd.concat([ETF_target,index_table],axis = 1)
    
    # move the target feature to the last column
    t = ETF_target['close'].values
    ETF_target.drop(['close'],axis=1,inplace = True)
    ETF_target['close'] = t
    
    return ETF_target


def load_ETFindex(fname,stock):
    print('load',fname)
    ETF_propotion = pd.read_excel(fname)
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


def load_orderbook(fname,ETF):
    print('load',fname)
    orderbook_ = pd.read_hdf('twse_orderbook_comp.h5').rename(columns = {'時間':'date',
                                                                    '累積委託買進筆數':'ask_vol',
                                                                    '累積委託買進數量':'ask_count',
                                                                    '累積委託賣出筆數':'bid_vol',
                                                                    '累積委託賣出數量':'bid_count',
                                                                    '累積成交筆數':'vol',
                                                                    '累積成交數量':'count',
                                                                    '累積成交金額':'amount'})
    #make the date index as same as ETF
    orderbook = orderbook_[orderbook_.index==0].iloc[-1*ETF.shape[0]-1:-1]
    orderbook['date'] = ETF.index.values
    orderbook = orderbook.groupby('date').sum().drop(['vol','count','amount'],axis = 1)
    
    return orderbook

