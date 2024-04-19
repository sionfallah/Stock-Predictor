import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt

model = load_model("Stock.predictions Model.keras")

st.header('Stock Market Predictor')

#Get user input for Stock name and stock price
stock = st.text_input('Enter Stock Symbol', 'GOOG')
start = st.date_input('Select Start Date', value=pd.to_datetime('2012-01-01'))
end = st.date_input('Select End Date', value=pd.to_datetime('2022-12-31'))

data = yf.download(stock, start, end)

st.subheader('Stock Data')
st.write(data)

data_train = pd.DataFrame(data.Close[0: int(len(data)*0.8)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.8): len(data)])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))


pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index= True)
data_test_scale = scaler.fit_transform(data_test)

st.subheader('Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'red')
plt.plot(data.Close, 'green')
plt.xlabel('Time (Years)')
plt.ylabel('Price ($)')
plt.show()
st.pyplot(fig1)

st.subheader('Price MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'red')
plt.plot(ma_100_days, 'blue')
plt.plot(data.Close, 'green')
plt.xlabel('Time (Years)')
plt.ylabel('Price ($)')
plt.show()
st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'red')
plt.plot(ma_100_days, 'blue')
plt.plot(data.Close, 'green')
plt.xlabel('Time (Years)')
plt.ylabel('Price ($)')
plt.show()
st.pyplot(fig3)

x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])
x, y = np.array(x), np.array(y)

predict = model.predict(x)

scale = 1/scaler.scale_

predict = predict * scale
y = y*scale

st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8,6))
plt.plot(predict, 'red', label = 'Original Price')
plt.plot(y, 'green', label = 'Predicted Price')
plt.xlabel('Time (Days)')
plt.ylabel('Price ($)')
plt.show()
st.pyplot(fig4)