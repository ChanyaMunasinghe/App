
import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

st.title("Real-Time Stock Prediction App")
stocks = ['AAPL', 'TSLA', 'GOOG']
selected_stock = st.selectbox("Select a stock", stocks)

@st.cache_data
def get_data(stock):
    df = yf.download(stock, period="2y")
    return df

def create_sequences(data, seq_length=60):
    X = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
    return np.array(X)

df = get_data(selected_stock)
st.line_chart(df['Close'])

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['Close']])
seq_length = 60
X = create_sequences(scaled_data)
X = X.reshape((X.shape[0], X.shape[1], 1))

rnn_model = load_model("rnn_model.h5")
lstm_model = load_model("lstm_model.h5")
rnn_pred = rnn_model.predict(X)
lstm_pred = lstm_model.predict(X)

rnn_pred = scaler.inverse_transform(rnn_pred)
lstm_pred = scaler.inverse_transform(lstm_pred)

st.subheader("Predicted Prices")
st.line_chart(rnn_pred[-100:], use_container_width=True)
st.line_chart(lstm_pred[-100:], use_container_width=True)
