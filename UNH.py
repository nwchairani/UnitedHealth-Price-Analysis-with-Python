"""
United Health 
@author: Novia Chairani

"""
import numpy as np
import pandas as pd
import scipy.stats as stats
import pandas_datareader.data as web
import datetime as dt
import matplotlib.pyplot as plt

#import machine learning packages
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix


###########################################################################################################
#Value at Risk - VaR#
###########################################################################################################

#historical data to approximate mean and standard deviation
start_date = dt.datetime(2016,1,1)
end_date = dt.datetime(2021,9,30)
	
#download citigroup stock related data from Yahoo Finance
unh = web.DataReader('UNH',data_source='yahoo',start=start_date,end=end_date)
	
#we can use pct_change() to calculate daily returns
unh['returns'] = unh['Adj Close'].pct_change()

#get the 0.05 quantile on the daily retuns distribution
unh['returns'].quantile(0.05)	

#calculate log daily returns
log_returns = np.log(unh['Adj Close']) - np.log(unh['Adj Close'].shift(1))
returns = np.log(unh['Adj Close']/(unh['Adj Close']).shift(1))

#get the 0.05 quantile on the log daily retuns distribution (empirical VaR)
var=log_returns.quantile(0.05)
print('Historical VaR: %0.3f' % var)

#histogram of log returns and plot the VaR
plt.figure(figsize=(10,6))
plt.hist(log_returns, bins=50)
plt.axvline(var, color='red', linestyle='dashed', linewidth=1)
#plt.text(var, 100, var.round(3),)
plt.annotate(var.round(3), xy=(var,100), xytext=(-.10, 100),
             arrowprops=dict(facecolor='red', shrink=0.01))

#we can assume daily returns to be normally distributed: mean and variance (standard deviation)
#can describe the process
#get the mean and standard deviation return
mu = np.mean(log_returns)
sigma = np.std(log_returns)

#get the 1 day VaR
z=stats.norm.ppf(0.05)
VaR=(sigma*z)
print('1 day Value at risk is: %0.3f' % VaR)

#get the 30 day VaR
n=30 #days
VaR30=(mu*n)+((sigma*np.sqrt(n))*z)
print('30 day Value at risk is: %0.3f' % VaR30)

#get the dollar amount VaR
S = 100000 #this is the investment (stocks or whatever) or (1e6=1000000)
dollarVaR30=S*VaR30
print('30 day Dollar Value at risk is: $%0.3f' % dollarVaR30)


###########################################################################################################
#Machine Learning
###########################################################################################################

#set lags
lags=2

#create a new dataframe
#we want to use additional features: lagged returns...today's returns, yesterday's returns etc
tslag = pd.DataFrame(index=unh.index)
tslag["Today"] = unh["Adj Close"]

# Create the shifted lag series of prior trading period close values
range(0, lags)
for i in range(0, lags):
    tslag["Lag%s" % str(i+1)] = unh["Adj Close"].shift(i+1)

#create the returns DataFrame
dfret = pd.DataFrame(index=tslag.index)
dfret["Today"] = tslag["Today"].pct_change()

#create the lagged percentage returns columns
for i in range(0, lags):
    dfret["Lag%s" % str(i+1)] = tslag["Lag%s" % str(i+1)].pct_change()
        
#because of the shifts there are NaN values ... we want to get rid of those NaNs
dfret.drop(dfret.index[:4], inplace=True)

#"Direction" column (+1 or -1) indicating an up/down day (0 indicates daily non mover)
dfret["Direction"] = np.sign(dfret["Today"])

#Replace where nonmover with down day (-1)
dfret["Direction"]=np.where(dfret["Direction"]==0, -1, dfret["Direction"] ) 

# Use the prior two days of returns as predictor 
# values, with todays return as a continuous response
x = dfret[["Lag1"]]
y = dfret[["Today"]]

"""
# Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)
"""

#Alternative test/train split
# The test data is split into two parts: Before and after 1st Jan 2018.
start_test = dt.datetime(2018,1,1)

# Create training and test sets
x_train = x[x.index < start_test]
x_test = x[x.index >= start_test]
y_train = y[y.index < start_test]
y_test = y[y.index >= start_test]


####################################
#Regression
####################################

#we use Decision Trees as the machine learning model
model=DecisionTreeRegressor(max_depth = 10)
#train the model on the training set
results=model.fit(x_train, y_train)

plt.figure(figsize=(22,16))
plot_tree(results, filled=True)

#make an array of predictions on the test set
y_pred = model.predict(x_test)
model.score(x_test, y_test)
#predict an example
x_example=[[0.01]]
yhat=model.predict(x_example)


#####################################
#Classification
#####################################

#plot log2 function (the measure of entropy)
plt.figure()
plt.plot(np.linspace(0.01,1),np.log2(np.linspace(0.01,1)))
plt.xlabel("P(x)")
plt.ylabel("log2(P(x))")
plt.show()

# Use the prior two days of returns as predictor 
# values, with direction as the discrete response
x = dfret[["Lag1","Lag2"]]
y = dfret["Direction"]

#Alternative test/train split
# The test data is split into two parts: Before and after 1st Jan 2018.
start_test = dt.datetime(2018,1,1)

# Create training and test sets
x_train = x[x.index < start_test]
x_test = x[x.index >= start_test]
y_train = y[y.index < start_test]
y_test = y[y.index >= start_test]


#we use Decision Trees as the machine learning model
model=DecisionTreeClassifier(criterion = 'entropy', max_depth = 19)
#train the model on the training set
results=model.fit(x_train, y_train)

#make an array of predictions on the test set
y_pred = model.predict(x_test)

#predict an example
x_example=[[0.01,0.01]]
yhat=model.predict(x_example)

#output the hit-rate and the confusion matrix for the model
print("Confusion matrix: \n%s" % confusion_matrix(y_pred, y_test))
print("Accuracy of decision tree model on test data: %0.3f" % model.score(x_test, y_test))

#plot decision tree
dt_feature_names = list(x.columns)
dt_target_names = ['Sell','Buy'] #dt_target_names = [str(s) for s in y.unique()]
plt.figure(figsize=(22,16))
plot_tree(results, filled=True, feature_names=dt_feature_names, class_names=dt_target_names)
plt.show()

#pruning (choosing max_depth parameter using 5 fold cross validation)
depth = []
for i in range(1,20):
    model = DecisionTreeClassifier(criterion='entropy', max_depth=i)
    # Perform 5-fold cross validation k=5
    scores = cross_val_score(estimator=model, X=x_train, y=y_train, cv=5)
    depth.append((scores.mean(),i))
    
print(max(depth))






























