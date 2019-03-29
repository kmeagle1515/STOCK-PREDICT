import pandas as pd
import csv
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
data_frame = pd.read_csv('AUDJPY-2016-01.csv', names=['Symbol', 'Date_Time', 'Bid', 'Ask'],

 index_col=1, parse_dates=True)

#data_frame.head()

data_ask =  data_frame['Ask'].resample('5Min').ohlc()

data_bid =  data_frame['Bid'].resample('5Min').ohlc()

export_csv=data_ask.to_csv(r'C:\Users\Owner\Desktop\new.csv',index=None,header=True)

#data_bid.head()

#data_ask_bid=pd.concat([data_ask, data_bid], axis=1, keys=['Ask', 'Bid'])
data_frame = pd.read_csv('new.csv')
x_train=data_frame.iloc[1:499].values
y_train=data_frame.iloc[1:499,1].values
x_test=data_frame.iloc[500:1345].values
y_test=data_frame.iloc[500:1345,1].values

regressor=SVR(kernel='linear',degree=1)

#xtrain,xtest,ytrain,ytest=train_test_split(x,y)

regressor.fit(x_train,y_train)

pred=regressor.predict(x_test)
#export_csv=pred.to_csv(r'C:\Users\Owner\Desktop\new1.csv',index=None,header=True)
#print(regressor.score(x_test,y_test))
print(r2_score(y_test,pred))
print (pred)