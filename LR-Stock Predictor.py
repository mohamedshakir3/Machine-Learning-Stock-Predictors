
################################################################################
# Predicts historical closing price of a stock using a linear regression model #
################################################################################

####### Import and Configure Libraries #######
import math
import pandas_datareader as pdr
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
#################################################


def main():
    ticker = input("ticker: ")
    days = int(input('Number of days for perdiciton: '))
    df = getStockData(ticker)
    df = getClosePrice(df)
    print(df)
    df = getTarget(df,days)
    print(df)
    featureData = getFeatureData(df,days)
    print(featureData)
    targetData = getTargetData(df,days)
    print(targetData)
    x_train, x_test, y_train, y_test = train_test_split(featureData,targetData,test_size = 0.25)
    lr = LinearRegression().fit(x_train, y_train)
    future = df.drop(['Projected'], 1)[:-days]
    future = np.array(future.tail(days))
    lr_prediction = lr.predict(future)
    print(lr_prediction)
    Visualize(df,lr_prediction,featureData)

def getStockData(symbol):
    return pdr.DataReader(f'{symbol}', data_source='yahoo',start='2011-01-01', end='2022-01-01')

def getClosePrice(dataframe):
    return dataframe[['Close']]

def getTarget(dataframe,x):
    dataframe['Projected'] = dataframe[['Close']].shift(-x)
    return dataframe

def getFeatureData(dataframe, x):
    return np.array(dataframe.drop(['Projected'], 1))[:-x]

def getTargetData(dataframe, x):
    return np.array(dataframe['Projected'])[:-x]

def createTrainingSet(x, y, size = 0.25):
    return train_test_split(x,y, test_size = size)

def Visualize(dataframe,predictions,featureDataSet):
    realData = dataframe[featureDataSet.shape[0]:]
    realData['Predictions'] = predictions
    plt.figure(figsize = (16,8))
    plt.title('Linear Regression Model')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.plot(dataframe['Close'])
    plt.plot(realData[['Close', 'Predictions']])
    plt.legend(['Orig', 'Val', 'Predict'])
    plt.show()


if __name__ == '__main__':
    main()
