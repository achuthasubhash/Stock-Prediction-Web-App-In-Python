# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 09:40:37 2021

@author: DELL
"""
#https://thecleverprogrammer.com/2021/06/03/end-to-end-machine-learning-model/
from autots import AutoTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

sns.set()
plt.style.use('seaborn-whitegrid')

import streamlit as st
st.title("Future Price Prediction Model")
df = st.text_input("Let's Predict the Future Prices")
x=st.sidebar.selectbox('Select one symbol', ( 'AAPL', 'MSFT',"SPY",'WMT'))
import datetime
today = datetime.date.today()
before = today - datetime.timedelta(days=700)
start = st.sidebar.date_input('Start date', before)
end = st.sidebar.date_input('End date', today)
if start < end:
    st.sidebar.success('Start date: `%s`\n\nEnd date:`%s`' % (start, end))
else:
    st.sidebar.error('Error: End date must fall after start date.')

forecast_length=st.slider('forecast duration', min_value=0, max_value=10)

def load_data(ticker):
    data = yf.download(ticker, start,end)
    data.reset_index(inplace=True)
    return data
data = load_data(x)
model = AutoTS(forecast_length=forecast_length, frequency='infer', ensemble='simple', drop_data_older_than_periods=200)
model = model.fit(data, date_col='Date', value_col='Close', id_col=None)
prediction = model.predict()
forecast = prediction.forecast
st.write(forecast)
