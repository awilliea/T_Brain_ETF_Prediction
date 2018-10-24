# Data 
1. Yuanta50
2. Taiwan50
3. orderbook (day orderbook data for INDEX)
5. oil_price
6. bond(fiveyear + ten year)
7. TXF
8. EXF
9. FXF
10. E_F(EXF/FXF)
11. Nikkei
12. SP(S&P 500)
13. DJ(Dow Jones)
14. Russel
15. VIX
16. SOX
17. pcr(put call ratio)
18. FOI(foreign investor OI)
19. IOI(Investment Trust OI)
20. DOI(Dealer OI) 
21. inf(infaltion rate)
# My model for ETF prediction
### Contents of jupyter notebook
 - Regression model for Yunta50   
 - Regression model for 18 ETF
 - Classification model for 18 ETF
 - Display for .py
 - Exploration of Yunta50
### Contents of .py 
- **load.py(return Dataframe)**
	* load_stock(fname)
	* load_ETFtable(fname)
	* load_ETFindex(fname,stock)
	* load_ETF(ETFtable,ETFcode)
	* load_orderbook(fname,ETF) : load orderbook and make the date of it as same as the ETF
 
- **price_onedaymodel.py**
	* one_day_prediction_18(ETFtable,ETF_list,window,lamb): return ETF_score,ETF_model,ETF_features
	* one_day_prediction(ETF,window,lamb) : return score,model,ETF_features
	* feature_importance(model) :display the weight of every features graph
	* feature_importance_scatter(model,window,ETF_features) : display the weight of ETF_features graph(group by category)

     lamb : the coeficient of the regularization

 
- **price_fivedaymodel.py**
	* five_day_prediction_18(ETFtable,ETF_list,window,lamb): return ETF_week_score(total 18 ETF score numpy array)
	* five_day_prediction(ETF,window,lamb) : return week_score(1 ETF score numpy array)
	* show_week_score_18(ETF_score) : display ETF_week_score for the recent weeks
	* show_week_score(score): display week_score for the recent weeks


- **class_onedaymodel.py**
	* one_day_prediction_regression_18(ETFtable,ETF_list,window,lamb,interval): return 18 ETF score
	* one_day_prediction_regression(ETF,window,lamb,interval) : print average score
	* one_day_prediction_svm_18(ETFtable,ETF_list,window,interval,kernel,C,gamma,coef0) :print score array
	* one_day_prediction_svm(ETF,window,interval,kernel,C,gamma,coef0) :print average train score and test score
	
	kernel,C,gamma,coef0 : the same as the sklearn.svm
	
	interval : the range to define rise of fall


 
- **class_fivedaymodel_re.py**
	* five_day_prediction_18(ETFtable,ETF_list,window,lamb) :return ETF_week_score
	* five_day_prediction(ETF,window,lamb): return week_score

     Can use show_week_score_18 ,show_week_score from price_fivedaymodel.py to display the score  

 
- **submit.py**
	* submit_data(ETFtable,ETF_list,window,lamb) : generate Submission.csv

