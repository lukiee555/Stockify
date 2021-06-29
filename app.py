import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st

start = '2010-01-01'
end = '2020-12-31'

st.title('Stockify : The Ultimate Stock Trend Predictor')

user_input = st.text_input('Enter Stock Ticker', 'GOOGL')

df = data.DataReader(user_input, 'yahoo', start, end)

#Describing The Data
st.subheader('Data from 2010 - 2020')
st.write(df.describe())

#Visualisation
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing Prive vs Time Chart with 100MA and 200MA')
#Moving Average For 100 Days
movingAvg100 = df.Close.rolling(100).mean()
#Moving Average For 200 Days
movingAvg200 = df.Close.rolling(200).mean()

fig1 = plt.figure(figsize = (12,6))
plt.plot(df.Close, 'b')
plt.plot(movingAvg100, 'r', label = "Moving Average 100")
plt.plot(movingAvg200, 'g', label = "Moving Average 200")
st.pyplot(fig1)

 #Reseting Index as I don't want Date as Index Column
df = df.reset_index()
df = df.drop(['Date', 'Adj Close'], axis = 1)

#Spliting data into training and testing 
data_train = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_test = pd.DataFrame(df['Close'][int(len(df)*0.70) : int(len(df))])

#Scaling Down the Data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_train_array = scaler.fit_transform(data_train)

X_Train = []
Y_Train = []

for i in range(200, data_train_array.shape[0]):
    X_Train.append(data_train_array[i - 200: i])
    Y_Train.append(data_train_array[i, 0])

X_Train, Y_Train = np.array(X_Train), np.array(Y_Train)

#Load My Model
model = load_model('Stockify_LSTM_Model.h5')

#Testing the model I need Past 200 Days data
past_200_days = data_train.tail(100)
final_df = past_200_days.append(data_test, ignore_index=True)
input_data = scaler.fit_transform(final_df)

X_Test = []
Y_Test = []

for i in range(200, input_data.shape[0]):
    X_Test.append(input_data[i-200:i])
    Y_Test.append(input_data[i, 0])

X_Test, Y_Test = np.array(X_Test), np.array(Y_Test)

#Prediction
Y_pred = model.predict(X_Test)

#Scaling Up
scaler = scaler.scale_
scaler_factor = 1/scaler[0]
Y_pred = Y_pred * scaler_factor
Y_Test = Y_Test * scaler_factor


#Final Chart
st.subheader('Original vs Prediction')
fig2 = plt.figure(figsize = (12,6))
plt.plot(Y_Test, 'b', label = 'Original Price')
plt.plot(Y_pred, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
