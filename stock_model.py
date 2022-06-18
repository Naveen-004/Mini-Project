import datetime
import numpy as np
import pandas as pd
import streamlit as st
from datetime import date
import plotly.express as px
from nsepy import get_history
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

st.title("Stock Price Prediction")
symbol = st.sidebar.text_input("Enter the stock symbol:", "sbin").upper()
start_date = st.sidebar.date_input("Start date for Stock analysis", datetime.date(2019,1,1))
end_date = st.sidebar.date_input("End date for Stock analysis", date.today())
data = get_history(symbol=symbol, start=start_date, end=end_date)

st.write("Stock symbol:",'"',symbol,'"')
st.write('Data from', start_date, 'to', end_date)
st.write(data.describe())

st.subheader('Closing Price vs Date')
fig = plt.figure(figsize=(12,8))
plt.plot(data.Close)
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title(symbol)
plt.legend(['Closing Price'])
st.pyplot(fig)
 
st.subheader('100 days moving average')
ma100 = data.Close.rolling(window=100).mean()
fig = plt.figure(figsize=(12,8))
plt.plot(ma100, color='red')
plt.plot(data.Close, color='blue')
plt.title(symbol + ' Closing Price vs Date with 100 days moving average')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend(['100 days moving average', 'Closing Price'])
st.pyplot(fig)

st.subheader('200 days moving average')
ma100 = data.Close.rolling(window=100).mean()
ma200 = data.Close.rolling(window=200).mean()
fig = plt.figure(figsize=(12,8))
plt.plot(ma200, color='green')
plt.plot(ma100, color='red')
plt.plot(data.Close, color='blue')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title(symbol + ' Closing Price vs Date with 200 days moving average')
plt.legend(['200 days moving average','100 days moving average', 'Closing Price'])
st.pyplot(fig)

# # Function to get stock data from NSE
# @st.cache
# def get_stock_data(symbol, start_date, end_date):
#     data = get_history(symbol=symbol, start=start_date, end=end_date)
#     return data

# @st.cache
# def data(symbol, start_date, end_date):
#     df = get_stock_data(symbol, start_date, end_date)
#     df.reset_index(inplace=True)
#     df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
#     train_df = pd.DataFrame(df.Close[0:int(len(df.Close)*0.75)])
#     test_df = pd.DataFrame(df.Close[int(len(df.Close)*0.75):int(len(df))])
#     return train_df, test_df
# train_df , test_df = data(symbol, start_date, end_date)

# def data_preprocessing(train_df, test_df): 

#     scaler = MinMaxScaler(feature_range=(0, 1))
#     scaled_train = scaler.fit_transform(train_df)
#     X_train = []
#     y_train = []
#     for i in range(100, len(scaled_train)):
#         X_train + scaled_train[i-100 : i]
#         y_train + scaled_train[i, 0]
#         X_train, y_train = np.array(X_train), np.array(y_train)

#     past_100 = train_df.tail(100)
#     final_df = past_100.append(test_df, ignore_index=True)    
#     input_data = scaler.fit_transform(final_df)
#     X_test = []
#     y_test = []
#     for i in range(100, len(input_data)):
#         X_test + input_data[i-100 : i]
#         y_test + input_data[i, 0]
#     X_test, y_test = np.array(X_test), np.array(y_test)
#     scale = scaler.scale_
#     return X_train, y_train, X_test, y_test, scale 

# X_train, y_train, X_test, y_test, scale = data_preprocessing(train_df, test_df)
# st.write("X_train:", X_train)
# def model_build(X_train, y_train, X_test):
#     model = Sequential()
#     model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], 1))) # input_shape=(timesteps, features)
#     model.add(Dropout(0.2))
#     #layer 2
#     model.add(LSTM(units=60, activation='relu', return_sequences=True))
#     model.add(Dropout(0.3))
#     #layer 3
#     model.add(LSTM(units=80, activation='relu', return_sequences=True))
#     model.add(Dropout(0.4))
#     #layer 4
#     model.add(LSTM(units=120, activation='relu', return_sequences=False))
#     model.add(Dropout(0.5))
#     model.add(Dense(units = 120))
#     model.compile(optimizer='adam', loss='mse')
#     model.fit(X_train, y_train, epochs=50)
#     y_pred = model.predict(X_test)
#     return y_pred

