
############################################################################
# Predicts the historic closing prices of a stock using an                 #
# artificial reccurent deep-learning neural network Long Short Term Memory #
############################################################################
import math
import pandas_datareader as pdr
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
############################################################################

def main():
    ticker = input("ticker: ")
    df = getStockData(ticker)
    df = getClosePrice(df)
    ds = df.values
    trainingData = math.ceil(len(ds) * 0.8)
    scaler = MinMaxScaler(feature_range=(0,1))
    scaledData = scaler.fit_transform(ds)
    trainData = scaledData[0:trainingData, :]
    x_train = []
    y_train = []

    for i in range(60, len(trainData)):
        x_train.append(trainData[i-60:i,0])
        y_train.append(trainData[i,0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, return_sequences = True, input_shape = (x_train.shape[1],1)))
    model.add(LSTM(50, return_sequences = False))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer = 'adam', loss='mean_squared_error')

    model.fit(x_train, y_train, batch_size = 1, epochs=1)

    testData = scaledData[trainingData - 60: , :]
    x_test = []
    y_test = ds[trainingData:, :]

    for i in range(60, len(testData)):
        x_test.append(testData[i-60:i,0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    rmse = np.sqrt(np.mean(predictions - y_test)**2)

    train = df[:trainingData]
    valid = df[trainingData:]
    valid['Predictions'] = predictions

    plt.figure(figsize=(16,8))
    plt.title('LSTM Model Stock Predictor')
    plt.xlabel('Date', fontsize = 18)
    plt.ylabel('Close Price', fontsize = 18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predict'])
    #plt.show()

    print(valid.tail(10))
    print(rmse)

def getStockData(symbol):
    return pdr.DataReader(f'{symbol}', data_source='yahoo',start='2011-01-01', end='2022-01-01')

def getClosePrice(dataframe):
    return dataframe[['Close']]

if __name__ == '__main__':
    main()
