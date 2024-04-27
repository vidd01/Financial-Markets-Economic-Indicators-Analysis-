# Basic libraries
import streamlit as st
import pandas as pd
import numpy as np
import base64
from PIL import Image
from azure.cosmos import CosmosClient, exceptions
from stocknews import StockNews
from math import sqrt
## DateTime libraries
# from datetime import date as dt
import datetime
import time
import yfinance as yf
import pandas_datareader as pdr
from statsmodels.tsa.seasonal import seasonal_decompose
## Keras and SKLearn libraries for LSTM model
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM,Dropout
from keras import optimizers
from keras.callbacks import Callback
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score,mean_absolute_percentage_error
## For XGBoost model
import os
import xgboost as xgb
from xgboost import XGBRegressor
from finta import TA
# For FB-Prophet model
from prophet import Prophet
# Plotly libraries
import plotly.graph_objects as go
import plotly.express as px
# Create plot for actual and predicted values
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt


#Used in Tab2 - macro data

def get_data_from_cosmos(country, generate_chart=True):
    connection_string = 'https://capstonesosmosstorage.documents.azure.com:443/'
    ckey = '6nWf2nyesaeRndaId1iR4QGkFqzLxj72moSOGBxSM7GqVRWgT6qteISZMUeGYgb7eYpm6aLs6XhzACDbSXeLFg=='
    country_db_mapping = {'India':'IndiaDB','Canada':'CanadaDB','USA':'UnitedStatesDB'}
    db_name = country_db_mapping[country]
    db_cn_name = country
    url = connection_string
    key = ckey
    client = CosmosClient(url, credential=key)
    database = client.get_database_client(db_name)
    container = database.get_container_client(db_cn_name)
    if country == "India":
        data = get_data(container, india_selected=True)
    else:
        data = get_data(container, chart = generate_chart)
    return data


def get_data(container, india_selected=False, chart=True):
    id_partitionkey =  [['1','gdp', 'GDP', 'In Trillions $'],['2','inflation', 'Inflation', 'Units'] ,['3','bondRates', 'BondRates', 'Percentage'],['4','unemployment', 'Unemployment', 'Percentage'], ['5','bankRates', 'BankRates', 'Percentage']]
    if india_selected:
        id_partitionkey =  [['1','gdp', 'GDP', 'In Trillions $'],['2','inflation', 'Inflation', 'Units'] ,['3','bondRates', 'BondRates', 'Percentage'], ['5','bankRates', 'BankRates', 'Percentage']]
    count_container = 1
    df_list = []
    col1, col2 = st.columns(2)
    for pair in id_partitionkey:
        item_id, partition_key = pair[0],pair[1]
        item = container.read_item(item_id, partition_key)
        # Filter out keys that are not dates
        filtered_data = {key: value for key, value in item.items() if '-' in key}
        df =dict(sorted(filtered_data.items()))
        df = pd.DataFrame(df.items(), columns=['date', 'value'])
        # Create two columns for side-by-side display
        if chart:
            if count_container % 2 == 0:
                generate_chart(df, pair[2], col2, pair[3])
            else:
                generate_chart(df, pair[2], col1, pair[3])
            count_container += 1
        df_list.append(df)
    if not chart:
        # Just return US Bank rate data
        item = container.read_item('5', 'bankRates')
        filtered_data = {key: value for key, value in item.items() if '-' in key}
        df =dict(sorted(filtered_data.items()))
        df = pd.DataFrame(df.items(), columns=['date', 'value'])
        return df
    return df_list

def generate_chart(data, name, col, y_axis_title=''):
    with col:
        if y_axis_title:
            st.markdown(f"""<span><Strong style='font-size: 24px;'>{name}</Strong> ({y_axis_title})</span>""" , unsafe_allow_html=True)
        fig = go.Figure()
        fig = fig.add_trace(
            go.Scatter(
                x=data['date'],
                y=data['value'],
                mode='lines',
                line=dict(color='white')
            )
        )
        fig.update_layout(    
                xaxis=dict(tickfont=dict(size=20)),
                yaxis=dict(tickfont=dict(size=20)),
                font=dict(size=16),  # Update overall font size
                )
        st.plotly_chart(fig, use_container_width=True)

def load_data(file):
	df = pd.read_csv(file)
	return df

def fetch_index_data(country):
    tickers = {
        'India': '^NSEI',
        'Canada': '^GSPTSE',
        'USA': '^IXIC ^GSPC ^DJI'
    }

    selected_tickers = tickers.get(country).split()

    data = yf.download(selected_tickers, start=datetime.date.today() - datetime.timedelta(days=365), end=datetime.date.today(), group_by='ticker')

    return data

def get_start_end_dates(selected_duration):
    end_date = datetime.date.today()
    if selected_duration == "5 Year":
        start_date = end_date - datetime.timedelta(days=5*365)
    elif selected_duration == "7 Years":
        start_date = end_date - datetime.timedelta(days=7*365)  # Approximation
    elif selected_duration == "10 Years":
        start_date = end_date - datetime.timedelta(days=10*365)  # Approximation
    else:
        start_date = None
    return start_date, end_date

def create_dataset(X, y, time_step):
    Xs, ys = [], []
    for i in range(len(X) - time_step):
        v = X[i:(i + time_step)]
        Xs.append(v)
        ys.append(y[i + time_step])
    return np.array(Xs), np.array(ys)

def color_coded_sentiment(score):
    if score > 0.5:
        color = 'green'
    elif score < -0.5:
        color = 'red'
    else:
        color = 'yellow'
    return f'<span style="color:{color};">{score:.4f}</span>'

def fetch_index_dataa(ticker, period):
    return yf.download(ticker, period=period, interval='1mo')

def display_country_index(data, key, title, additional_data=False):
    st.write(f'<div style="text-align: center;"><h3>{title}</h3></div>', unsafe_allow_html=True)
    index_data = data['Adj Close'] if not additional_data else data[key]['Adj Close']
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=index_data.index, y=index_data.values, mode='lines', name=key))
    fig.update_layout(
        xaxis=dict(tickfont=dict(size=20)),
        yaxis=dict(tickfont=dict(size=20)),
        font=dict(size=16),
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Adding decomposition graph
    decomposition = seasonal_decompose(index_data.dropna(), model='additive', period=12)
    fig_dec = go.Figure()
    fig_dec.add_trace(go.Scatter(x=index_data.index, y=decomposition.trend, mode='lines', name='Trend'))
    fig_dec.add_trace(go.Scatter(x=index_data.index, y=decomposition.seasonal, mode='lines', name='Seasonal'))
    fig_dec.add_trace(go.Scatter(x=index_data.index, y=decomposition.resid, mode='lines', name='Residual'))
    fig_dec.update_layout(
        xaxis=dict(tickfont=dict(size=20)),
        yaxis=dict(tickfont=dict(size=20)),
        font=dict(size=16),
        title='Time Series Decomposition'
    )
    st.plotly_chart(fig_dec, use_container_width=True)

    st.markdown("---")

