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

## Keras and SKLearn libraries for LSTM model
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM,Dropout
from keras import optimizers
from keras.callbacks import EarlyStopping
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
# Create plot for actual and predicted values
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt


######################################### Page heading and image ########################################
st.set_page_config(page_title='Capstone 2 Web App', page_icon=':star:', 
                   layout="wide")

st.markdown('<h1><center>Financial Markets and Macro Economic Indicators</h1>', unsafe_allow_html=True)


def get_us_bank_rates():
    data = get_data_from_cosmos('USA',generate_chart=False)
    return data

def get_data(container, india_selected=False, chart=True):
    id_partitionkey =  [['1','gdp', 'GDP'],['2','inflation', 'Inflation'] ,['3','bondRates', 'BondRates'],['4','unemployment', 'Unemployment'], ['5','bankRates', 'BankRates']]
    if india_selected:
        id_partitionkey =  [['1','gdp', 'GDP'],['2','inflation', 'Inflation'] ,['3','bondRates', 'BondRates'], ['5','bankRates', 'BankRates']]
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
                generate_chart(df, pair[2], col2)
            else:
                generate_chart(df, pair[2], col1)
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

def generate_chart(data, name, col):
    with col:    
        st.subheader(name)
        fig = go.Figure()
        fig = fig.add_trace(
            go.Scatter(
                x=data['date'],
                y=data['value'],
                mode='lines',
                line=dict(color='white')
            )
        )
        st.plotly_chart(fig, use_container_width=True)

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


    
def main():
    image = Image.open("banner.jpg")

    col1, col2, col3 = st.columns([1,2,1])  # The middle column is twice the size of the side columns

    with col2:
        st.image(image)  # This will make the image fit the column width
        
    left, mid, right = st.columns([1,6,1])
    with mid:
        data_returned = tabs()

@st.cache_data
def load_data(file):
	df = pd.read_csv(file)
	return df

######################################### Define the tabs ########################################

def tabs():
    tab1, tab6, tab2, tab3, tab4, tab5 = st.tabs([":black_square_button: Application Overview", ":black_square_button: Financial Market Indices",":black_square_button: Macro-Economic Indicators", ":black_square_button: Correlation Matrix - Heatmap", ":black_square_button: Machine Learning based Price Forecast ", ":black_square_button: Company News and Sentiment Analysis"])

######################################### Overview Tab ########################################    
    with tab1:
        st.subheader('Financial Markets & Economic Indicators of USA | Canada | India')
        st.markdown("""
            Explore financial markets and economic indicators of USA, Canada, and India. Analyze correlations, forecast ETF prices, and gain insights from news sentiment.
            * <b>MACRO-ECONOMIC INDICATORS DASHBOARD:</b> Explore various economic indicators for the United States, Canada, and India. You can likely filter by country and visualize trends over time using charts and graphs.
            * <b>CORRELATION MATRIX:</b> Visual representation of the relationships between different economic indicators and financial assets
            * <b>SENTIMENT ANALYSIS:</b> This section analyzes news articles and other sources to gauge the overall sentiment surrounding the financial markets. This can be helpful in understanding how current events might be impacting investor confidence and market behavior.                 
        """, unsafe_allow_html=True)

        idx_info = load_data("etfs.csv")

        st.write('## ETF Description')
        with st.expander('CLICK HERE FOR MORE INFORMATION'):
            st.table(idx_info)

######################################### MacroEconomic Indicators ########################################
    with tab2:
        st.write('## Macro Economic  Historical Data')
        with st.form(key='country_selection_form'):
            country = st.selectbox('Select Country', ['India', 'Canada', 'USA'])
            if st.form_submit_button(label='Submit'):
                data = get_data_from_cosmos(country)

############################################ Correlation Matrix ############################################
    with tab3:
        st.write("Correlation Analysis")

############################################ ML Modelling####################################################
    with tab4:
        stock_name = st.text_input('Enter Stock ticker', '')
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

    # If stock name is entered, show the date input
        if stock_name:
            
            # Add Multiple date selection option to the select box
            user_date = st.selectbox('Select Duration', ["5 Year", "7 Years", "10 Years", "Custom Date"])
            if user_date == "Custom Date":
                start_date = st.date_input("Select Start Date", value=datetime.date.today() - datetime.timedelta(days=5*365), min_value=datetime.date(1900, 1, 1), max_value=datetime.date.today())
                end_date = datetime.date.today()
            else:
                start_date, end_date = get_start_end_dates(user_date)

        # If a start date is selected, show the model selection box
            if user_date:
                model_selection = st.selectbox(
                    "Select the prediction model",
                    options=["XGBoost", "LSTM", "FB-Prophet"])

                #Down the stock data based on user selection
                end_date = datetime.date.today()
                symbol = stock_name
                data = yf.download(symbol, start=start_date, end=end_date)
                # if isinstance(data.index, pd.DatetimeIndex):

                # Download the Bank Interest data from FRED
                series = 'FEDFUNDS'
                df_bank = pdr.get_data_fred(series, start_date, end_date)
                df_bank.reset_index(inplace = True)
                df_bank['DATE'] = pd.to_datetime(df_bank['DATE']).dt.strftime('%Y-%m-%d')

                df_bank= df_bank.rename(columns={"DATE":"Date","FEDFUNDS":'Bank_Interest'})
                df_bank = df_bank.set_index('Date',drop=True)
                df_bank_daily = df_bank.copy()
                # Update the Bank Interest rate data from monthly to daily data, to be in sync with the ETF data.
                df_bank_daily.index = pd.to_datetime(df_bank_daily.index)
                df_bank_daily = df_bank_daily.resample('D').ffill()

                # Join the stock and interest data
                # The Bank Interest rate column for the current month will be dislayed as NaN. since the data for current month is not available.
                df_combined = data.join(df_bank_daily,how='left')

                # We will Forward Fill the Bank Interest column for the current month with the available data from the previous month
                df_combined['Bank_Interest'] = df_combined['Bank_Interest'].ffill()

                df_combined['Prev Adj Close'] = df_combined['Adj Close'].shift(1)
                df_combined = df_combined.dropna()

########################## The below code will specific to each model ######################################
                
                ################ If model selected is LSTM:#################

                if model_selection == "LSTM":
                    # Features and Target
                    features = df_combined[['Prev Adj Close', 'Bank_Interest']]  # Including Previous Day's Adj Close and Interest
                    target = df_combined['Adj Close']

                    # Scaling the features and target
                    scaler_features = MinMaxScaler(feature_range=(0, 1))
                    scaled_features = scaler_features.fit_transform(features)
                    scaler_target = MinMaxScaler(feature_range=(0, 1))
                    scaled_target = scaler_target.fit_transform(target.values.reshape(-1, 1))

                    # Function to create dataset for LSTM
                    def create_dataset(X, y, time_step):
                        Xs, ys = [], []
                        for i in range(len(X) - time_step):
                            v = X[i:(i + time_step)]
                            Xs.append(v)
                            ys.append(y[i + time_step])
                        return np.array(Xs), np.array(ys)

                    time_steps = 30  # Using 30 days previous timestep
                    X, y = create_dataset(scaled_features, scaled_target, time_steps)

                    # Splitting the dataset into training and test sets
                    split = int(len(X) * 0.8)  # 80% for training, 20% for testing
                    X_train, X_test = X[:split], X[split:]
                    y_train, y_test = y[:split], y[split:]

                    # LSTM Model
                    model = Sequential([
                        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
                        LSTM(50),
                        Dense(1)
                    ])


                    # Compile the model
                    model.compile(optimizer='adam', loss='mean_squared_error')

                    # Train the model
                    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)
                    # Prediction
                    y_pred = model.predict(X_test)

                    # Inverse scaling for plotting and evaluation
                    y_test_inv = scaler_target.inverse_transform(y_test)
                    y_pred_inv = scaler_target.inverse_transform(y_pred)

                    ## Calculate the error metrics
                    mae = mean_absolute_error(y_test_inv, y_pred_inv)
                    mse = mean_squared_error(y_test_inv, y_pred_inv)
                    rmse = np.sqrt(mse)
                    mape = mean_absolute_percentage_error(y_test_inv, y_pred_inv)
                    r_square = r2_score(y_test_inv, y_pred_inv)


                    ## # Plot the Actual vs Predicted Price
                    test_dates = df_combined.index[split + time_steps:] # get the data of test data for the chart

                    # Ensure the length of test_dates matches y_test and y_pred arrays length
                    # This step is necessary as the last (time_steps-1) dates will not have corresponding predictions
                    test_dates = test_dates[:len(y_test)]

                    # Convert the date to a format suitable for plotting
                    test_dates_str = test_dates.strftime('%Y-%m-%d')

                    # Create traces with dates on the x-axis
                    chart1 = go.Scatter(
                        x=test_dates_str,
                        y=y_test_inv.flatten(),
                        mode='lines',
                        name='Actual Price'
                    )
                    chart2 = go.Scatter(
                        x=test_dates_str,
                        y=y_pred_inv.flatten(),
                        mode='lines',
                        name='Predicted Price'
                    )

                    # Layout with center-aligned title
                    layout = go.Layout(
                        title={
                            'text': f'Actual vs Forecasted Adj. Close Price of {symbol}',
                            'x': 0.5,  # Center title
                            'xanchor': 'center'
                        },
                        xaxis_title='Date',
                        yaxis_title='Adjusted Close Price',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        width=1000,
                        height=600
                    )

                    # Figure
                    fig_lstm = go.Figure(data=[chart1, chart2], layout=layout)

                    st.plotly_chart(fig_lstm)
                    
                    # Display the error metrics in the app as a table
                    metrics_lstm = {
                        "Name": ["Mean Absolute Error(MAE)", "Mean Squared Error(MSE)", "Root Mean Squared Error(RMSE)", "Mean Absolute Percentage Error(MAPE)", "R-squared value(R^2)"],
                        "Value": [mae, mse, rmse, mape, r_square]
                    }

                    # Convert to DataFrame
                    metrics_lstm_df = pd.DataFrame(metrics_lstm)

                    st.subheader('Error Metrics of the LSTM Model:')

                    # Display as a table in Streamlit
                    st.table(metrics_lstm_df)
                
                ################## If model selected is XGBoost:########################

                if model_selection == "XGBoost":
                    
                    st.markdown("XGBoost Model can be used to forecast the next single period price")
                    
                    # Get the data for the model
                    df_xgb = df_combined.copy()

                    # create different statistical metrics for the model.
                    df_xgb['EMA_9'] = df_xgb['Adj Close'].ewm(9).mean().shift()
                    df_xgb['SMA_5'] = df_xgb['Adj Close'].rolling(5).mean().shift()
                    df_xgb['SMA_10'] = df_xgb['Adj Close'].rolling(10).mean().shift()
                    df_xgb['SMA_15'] = df_xgb['Adj Close'].rolling(15).mean().shift()
                    df_xgb['SMA_30'] = df_xgb['Adj Close'].rolling(30).mean().shift()

                    # Calculate the Technical analysis value for the dataset.
                    df_xgb['RSI'] = TA.RSI(df_xgb)
                    df_xgb['ATR'] = TA.ATR(df_xgb)
                    df_xgb['BBWidth'] = TA.BBWIDTH(df_xgb)
                    df_xgb['Williams'] = TA.WILLIAMS(df_xgb)

                    # Create a new Target column which is the Adj. Close price for next day
                    df_xgb['Target'] = df_xgb['Adj Close'].shift(-1)

                    # dropping the N/A datas from the dataframe.
                    df_xgb = df_xgb.dropna()

                    # Select only the necessary columns for our model
                    df_xgb_final = df_xgb.drop(['Open','High','Low','Close','Volume','Prev Adj Close'],axis=1)

                    # create function for splitting the data into train and test
                    def data_split(data,perc):
                        data = data.values
                        n = int(len(data)*(1-perc))
                        return data[:n],data[n:]
                    
                    # splitting the data
                    train,test = data_split(df_xgb_final,0.20)

                    # Getting the data ready for training the model
                    X = train[:,:-1]
                    y = train[:,-1]

                    # fit the model with the training data and hyperparameters
                    model_xgb = XGBRegressor(objective='reg:squarederror', n_estimators=350, learning_rate=0.05, colsample_bytree=0.7,max_depth=3,gamma=5)
                    model_xgb.fit(X,y)

                    

                    #Create function for training the model and predicting a single value at a time.

                    def xgb_predict(train, val):
                        train = np.array(train)
                        X, y = train[:, :-1], train[:,-1]
                        model = XGBRegressor(objective='reg:squarederror', n_estimators=350, learning_rate=0.05, colsample_bytree=0.7,max_depth=3,gamma=5)
                        model.fit(X,y)
                        val = np.array(val).reshape(1, -1)
                        pred = model.predict(val)
                        return pred[0]
                    
                    # This function is for the overall model which utilizes the other functions for predictions and calculating the error metrics.

                    def validate(data, perc):
                        predictions = []
                        train, test = data_split(data, perc)
                        history = [x for x in train]
                        st.markdown('Training the model and forecasting the price. Please wait 	:hourglass_flowing_sand:')
                        progress_bar = st.progress(0)

                        for i in range(len(test)):
                            percent_complete = int(100 * (i + 1) / len(test))
                            progress_bar.progress(percent_complete)
                            
                            X_test, y_test = test[i, :-1], test[i, -1]
                            pred = xgb_predict(history, X_test)
                            predictions.append(pred)

                            history.append(test[i])
                            
                        progress_bar.empty()  # Hide the progress bar after completion
                        st.success('Forecasting complete!')

                        mse = mean_squared_error(test[:, -1], predictions)
                        rmse = mean_squared_error(test[:, -1], predictions, squared=False)
                        mae = mean_absolute_error(test[:, -1], predictions)
                        mape = np.mean(np.abs((test[:, -1]-predictions)/test[:, -1]))
                        # MAPE = mean_abs(test[:,-1], predictions)
                        r_sq = r2_score(test[:,-1], predictions)
                        return mse,rmse,mae,mape,r_sq, test[:, -1], predictions
                    
                    ## perform model predictions
                    mse,rmse,mae,mape,r_square, y, pred = validate(df_xgb_final, 0.2)

                    # Add test and pred array.
                    pred = np.array(pred)
                    test_pred = np.c_[test,pred]

                    ## Combine the test dataset with the predicted values to create a dataframe

                    df_final = pd.DataFrame(test_pred,columns=['Adj Close','Bank_Interest','EMA_9','SMA_5','SMA_10','SMA_15','SMA_30','RSI','ATR','BBWidth','Williams','Target','Predicted'])
                    # st.write(df_final)
                    

                    fig_xgb_f = make_subplots(rows=1, cols=1)

                    fig_xgb_f.add_trace(go.Scatter(x=df_xgb_final[len(train):].index, y=df_final['Target'], name="Actual Price"), row=1, col=1)
                    fig_xgb_f.add_trace(go.Scatter(x=df_xgb_final[len(train):].index, y=df_final['Predicted'], name="Predicted Price"), row=1, col=1)

                    # Update layout with titles and axis labels
                    fig_xgb_f.update_layout(
                        title_text=f'Actual vs Forecasted Adj. Close Price of {symbol}',
                        title_font_size=18,
                        title_x=0.3,
                        xaxis_title='Date',
                        yaxis_title='Price',
                        xaxis=dict(titlefont=dict(size=14)),
                        yaxis=dict(titlefont=dict(size=14)),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        width=1000,
                        height=600
                    )

                    # Show the plot in Streamlit
                    st.plotly_chart(fig_xgb_f)
                    st.markdown(f"The Adjusted Close Price on {df_combined.tail(1).index.date[0].strftime('%Y-%m-%d')} is {df_final.tail(1)['Target'].values[0]:.2f} and the model's prediction was {df_final.tail(1)['Predicted'].values[0]:.2f}")

                    # Display the error metrics in the app as a table
                    metrics_xgb = {
                        "Name": ["Mean Absolute Error(MAE)", "Mean Squared Error(MSE)", "Root Mean Squared Error(RMSE)", "Mean Absolute Percentage Error(MAPE)", "R-squared value(R^2)"],
                        "Value": [mae, mse, rmse, mape, r_square]
                    }
                    
                    # Convert to DataFrame
                    metrics_xgb_df = pd.DataFrame(metrics_xgb)

                    st.subheader('Error Metrics of the XGBoost Model:')

                    # Display as a table in Streamlit
                    st.table(metrics_xgb_df)

                ################## If model selected is FB-Prophet:########################
                
                if model_selection == "FB-Prophet":
                    month = st.slider('Set the no. of month for prediction', 1, 100, 12)
                    period = month*30
                    df_prophet = df_combined.copy()

                    # Reset the index of the dataframe to make Date as a column
                    df_prophet = df_prophet.reset_index()

                    # Prepare the dataset for Prophet with two columns: "ds" and "y"
                    prophet_df = df_prophet.rename(columns={'Date': 'ds', 'Adj Close': 'y'})

                    # Initialize the Prophet model and add 'Bank_Interest' as an additional regressor
                    model_fbp = Prophet(daily_seasonality=False, yearly_seasonality=True)
                    model_fbp.add_regressor('Bank_Interest')

                    # Fit the model
                    model_fbp.fit(prophet_df)

                    # Make a future dataframe for predictions
                    future = model_fbp.make_future_dataframe(periods=period)

                    # Update the future values of the regressor 'Bank_Interest'. 
                    # Since we don't have future values of 'Bank_Interest', we'll use the last available value for our scenario.

                    future['Bank_Interest'] = df_prophet['Bank_Interest'].iloc[-1]

                    # Predict
                    forecast = model_fbp.predict(future)

                    # Calculate and print the metrics
                    y_true = prophet_df['y']
                    y_pred = forecast['yhat'][:len(y_true)]

                    mae = mean_absolute_error(y_true, y_pred)
                    mse = mean_squared_error(y_true, y_pred)
                    rmse = np.sqrt(mse)
                    mape = mean_absolute_percentage_error(y_true, y_pred)
                    r_square = r2_score(y_true, y_pred)

                    # Plotting the actual and forecasted values
                    fig_fb = go.Figure()

                    # Actual data
                    fig_fb.add_trace(go.Scatter(x=prophet_df['ds'], y=prophet_df['y'], name='Actual Price'))

                    # Forecasted data
                    fig_fb.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="Predicted Price"))

                    # Plot components
                    fig_fb.update_layout(title={
                            'text': f'Actual vs Forecasted Adj. Close Price of {symbol}',
                            'x': 0.5,  # Center title
                            'xanchor': 'center'
                        },
                        xaxis_title='Date',
                        yaxis_title='Adjusted Close Price',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        width=1000, height=600)

                    # Show the plot in Streamlit
                    st.plotly_chart(fig_fb)

                    # Display the error metrics in the app as a table
                    metrics_fb = {
                        "Name": ["Mean Absolute Error(MAE)", "Mean Squared Error(MSE)", "Root Mean Squared Error(RMSE)", "Mean Absolute Percentage Error(MAPE)", "R-squared value(R^2)"],
                        "Value": [mae, mse, rmse, mape, r_square]
                    }
                    
                    # Convert to DataFrame
                    metrics_fb_df = pd.DataFrame(metrics_fb)

                    st.subheader('Error Metrics of the FB-Prophet Model:')

                    # Display as a table in Streamlit
                    st.table(metrics_fb_df)

                    # Plot the components of FB-Prophet
                    # Extract and plot the model components

                    st.subheader('FB-Prophet model components:')
                    trend = forecast[['ds', 'trend']]
                    yearly = forecast[['ds', 'yearly']]
                    weekly = forecast[['ds', 'weekly']]

                    # Plot Trend Component
                    fig_trend = go.Figure()
                    fig_trend.add_trace(go.Scatter(x=trend['ds'], y=trend['trend'], mode='lines', name='Trend'))
                    fig_trend.update_layout(title={
                                        'text': 'Trend Component',
                                        'x': 0.5,  # Center align the title
                                        'xanchor': 'center'
                                        },
                                        xaxis_title='Date',
                                        yaxis_title='Trend',
                                        width=1000,
                                        height=400
                                )                    
                    # Show the plot in Streamlit
                    st.plotly_chart(fig_trend)

                    # Plot Yearly Seasonality Component
                    fig_yearly = go.Figure()
                    fig_yearly.add_trace(go.Scatter(x=yearly['ds'], y=yearly['yearly'], mode='lines', name='Yearly Seasonality'))
                    fig_yearly.update_layout(title={
                                        'text': 'Yearly Seasonality Component',
                                        'x': 0.5,  # Center align the title
                                        'xanchor': 'center'
                                        },
                                        xaxis_title='Date',
                                        yaxis_title='Yearly Seasonality',
                                        width=1000,
                                        height=400
                                )                    
                    # Show the plot in Streamlit
                    st.plotly_chart(fig_yearly)

################## The below code is for the Sentiment Analysis #####################################################
    # Create color-coded sentiment indicators
    def color_coded_sentiment(score):
        if score > 0.5:
            color = 'green'
        elif score < -0.5:
            color = 'red'
        else:
            color = 'yellow'
        return f'<span style="color:{color};">{score:.4f}</span>'


    with tab5:
        stock_sentiment_ticker=st.text_input('Name of Stock',value="")
        
        if stock_sentiment_ticker:
                          
            # st.write("This is the closing price prediction tab.")
            st.subheader(f'News of {stock_sentiment_ticker}')
            sn=StockNews(stock_sentiment_ticker,save_news=False)
            df_news= sn.read_rss()
            st.write(df_news)
            df_news['published'] = pd.to_datetime(df_news['published']).dt.strftime("%Y-%m-%d %H:%M")
            # df['hour_minute'] = df['date_column']
            # st.write(df_news)
            # st.write(df_news.published.duplicated().sum())
            for i in range(10):
                # st.subheader(f'News {i+1}')
                st.write(f"<b>{df_news['title'][i]} - </b><small>{df_news['published'][i]}</small>",unsafe_allow_html=True)
                # st.write(str(i),df_news['published'][i])
                # st.write(df_news['title'][i])
                st.caption(df_news['summary'][i])
                title_sentiment= df_news['sentiment_title'][i]
                news_sentiment= df_news['sentiment_summary'][i]
                st.write(f'<b>Title Sentiment: {color_coded_sentiment(title_sentiment)} | \
                        News Sentiment: {color_coded_sentiment(news_sentiment)}<b>',unsafe_allow_html=True)
                # st.write(f'News Sentiment:', color_coded_sentiment(news_sentiment),unsafe_allow_html=True)
                st.markdown('---')

if __name__ == '__main__':
    main()
