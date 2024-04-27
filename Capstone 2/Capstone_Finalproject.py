from func import *

st.set_page_config(page_title='Capstone Project ', page_icon=':star:', 
                   layout="wide")

def main():
    image = Image.open("banner.jpg")

    col1, col2, col3 = st.columns([1,2,1])  # The middle column is twice the size of the side columns

    with col2:
        st.image(image)  # This will make the image fit the column width
        
    left, mid, right = st.columns([1,6,1])
    with mid:
        data_returned = tabs()

st.markdown("""
            <div style='text-align: center;'>
                <h3>Financial Markets and Macro-Economic indicators of USA, Canada, and India</h3> 
            </div>
        """, unsafe_allow_html=True)

######################################### Define the tabs ########################################

def tabs():
    tab1, tab6, tab2, tab3, tab4, tab5 = st.tabs(["App - Overview", "Financial Market Indices","Macro-Economic Indicators", "Correlation Matrix - Heatmap", "Machine Learning - Price Forecast ", "Company News and Sentiment Analysis"])

######################################### Overview Tab ########################################    
    with tab1: #Checked content
        # st.subheader('Financial Markets & Economic Indicators',divider=True)
        st.markdown("""
            <h2><strong>Analyze time series trends, visualize data, check for correlations, forecast prices using machine learning, and gain insights from news sentiment analysis.</strong></h2>
            <ul>
                <li><strong><h4>Financial Market Indices:</h4></strong> This tab provides an overview of major financial market indices such as the Dow Jones, S&P 500, NASDAQ, and others. Users can view real-time data, historical charts, and trends that help in understanding the current market conditions. This section serves as a pulse check for the broader market, offering insights into overall economic health and investor sentiment.</li>
                <li><strong><h4>Macro-Economic Indicators:</h4></strong> Focus on key macro-economic indicators such as Gross Domestic Product (GDP), inflation, unemployment rates, and others that influence financial markets and investment decisions. This tab allows users to explore these indicators over time and across different geographies, providing a macroeconomic context that affects asset prices and market movements.</li>
                <li><strong><h4>Correlation Matrix - Heatmap:</h4></strong> This tab features a heatmap of correlation matrices that show the relationship between different indices or macro-economic indicators. It helps users identify which variables move together, which are inversely related, and which are unrelated. This tool is invaluable for risk management and portfolio diversification, as it helps in understanding the co-movements of assets.</li>
                <li><strong><h4>Machine Learning based Price Forecast:</h4></strong> Here, users can access predictions for asset prices based on machine learning models. This tab features models like XGBoost, LSTM, and other time series models including analyze historical data to forecast future prices. It provides a data-driven approach to estimate future market behavior, which can be essential for planning investment strategies.</li>
                <li><strong><h4>Company News and Sentiment Analysis:</h4></strong> This section aggregates recent news articles related to specific companies or the overall market. It includes sentiment analysis to gauge the tone and sentiment of the news (positive, neutral, or negative) and its potential impact on the stock prices. This helps investors stay updated with the latest developments and understand how these could affect their investment decisions.</li>
            </h4></ul>
        """, unsafe_allow_html=True)


        macro_info,fin_info,etf_info = load_data(r"macro_info.csv"),load_data(r"fin_info.csv"),load_data(r"etf.csv")

        st.write('## Key Variable Description')
        with st.expander('Click here for know Macro Economic Indicators'):
            st.table(macro_info)
        with st.expander('Click here to know about Financial Indices'):
            st.table(fin_info)
        with st.expander('Click here to know about ETF tickers'):
            st.table(etf_info)
        
##########################################################
    with tab6:
        st.write('<style>div.Widget.row-widget.stPlotlyChart {display: flex; justify-content: center;}</style>', unsafe_allow_html=True)
        st.write('## Financial Market Indices Historical Data')
        st.write("<h3>Select Country</h3>", unsafe_allow_html=True)
        country = st.selectbox('', ['USA', 'Canada', 'India'])

        st.write("<h3>Select Time Period</h3>", unsafe_allow_html=True)
        time_period = st.selectbox('Choose period:', ['3y', '5y', '10y'])

        if country == 'India':
            ticker = '^NSEI'  # Nifty 50 ticker
        elif country == 'Canada':
            ticker = '^GSPTSE'  # S&P/TSX Composite Index ticker
        else:  # USA
            ticker = '^IXIC'  # NASDAQ Composite Index ticker

        data = fetch_index_dataa(ticker, time_period)

        if country == 'India':
            display_country_index(data, 'NIFTY 50', 'NIFTY 50 - Adjusted Close Price')
        elif country == 'Canada':
            display_country_index(data, 'TSX', 'S&P/TSX Composite index - Adjusted Close Price')
        elif country == 'USA':
            display_country_index(data, 'NASDAQ', 'NASDAQ - Adjusted Close Price')
            display_country_index(data, '^DJI', 'Dow Jones - Adjusted Close Price')
            display_country_index(data, '^GSPC', 'S&P 500 - Adjusted Close Price')
                    
######################################### MacroEconomic Indicators ########################################
    with tab2:
        st.write('## Macro Economic  Historical Data')
        st.write("<h3>Select Country</h3>", unsafe_allow_html=True)
        with st.form(key='country_selection_form'):
            country = st.selectbox('', ['USA','Canada','India'])
            if st.form_submit_button(label='Submit'):
                data = get_data_from_cosmos(country)

############################################ Correlation Matrix ############################################
    with tab3:
        import seaborn as sns
        st.write("Correlation Analysis")
        
        def load_data_cm(country,yearz):
            # Define the end date as today and start date as fifteen years from today
            end_date = pd.Timestamp.today()
            start_date = end_date - pd.DateOffset(years=yearz)

            if country == 'USA':
                fred_series = ['GDPA', 'CPIAUCSL', 'FEDFUNDS', 'IRLTLT01USM156N']
                indices = ['^GSPC', '^DJI', '^IXIC']
                labels = ['USA GDP', 'USA Inflation', 'USA Bank Rate', 'USA Bond Rate', 'S&P 500', 'Dow Jones', 'Nasdaq']

            elif country == 'Canada':
                fred_series = ['NGDPRXDCCAA', 'IRLTLT01CAM156N', 'CPALCY01CAM661N', 'IR3TIB01CAM156N']
                indices = ['^GSPTSE']
                labels = ['Canada GDP', 'Canada Bond Rate', 'Canada Inflation', 'Canada Bank Rate', 'TSX']

            elif country == 'India':
                fred_series = ['MKTGDPINA646NWDB', 'INDIRLTLT01STM', 'INDCPIALLMINMEI', 'INDIR3TIB01STM']
                indices = ['^NSEI']
                labels = ['India GDP', 'India Bond Rate', 'India Inflation', 'India Bank Rate', 'Nifty 50']

            # Fetch data from FRED
            fred_data = pdr.get_data_fred(fred_series, start_date, end_date)
            fred_data = fred_data.ffill()  # Forward fill for all FRED data
            fred_data.columns = labels[:len(fred_series)]  # Rename FRED data columns

            # Fetch indices data using Yahoo Finance
            indices_data = yf.download(indices, start=start_date, end=end_date, interval='1mo')['Adj Close']
            if len(indices) == 1:  # Handle single index series to prevent errors
                indices_data = indices_data.to_frame()
                indices_data.columns = [labels[len(fred_series):][0]]
            else:
                indices_data.columns = labels[len(fred_series):]  # Rename columns for indices

            # Combine the data into a single DataFrame and drop NaN values
            combined_data = pd.concat([fred_data, indices_data], axis=1).dropna()
            

            return combined_data

        # Streamlit user input
        country = st.selectbox('Select Country', ['USA', 'Canada', 'India'])
        yearz = st.slider('Number of Years', 1, 40, 15)
        data_cm = load_data_cm(country,yearz=yearz)

        # Selection of data series for correlation analysis
        selected_features = st.multiselect('Select data series for correlation:', data_cm.columns)

        # Display correlation matrix using Plotly
        if len(selected_features) >= 2:
            correlation_matrix = data_cm[selected_features].corr().round(2)
            fig = px.imshow(correlation_matrix, text_auto=True, aspect="auto", 
                            labels=dict(color="Correlation"),
                            x=selected_features, y=selected_features,
                            color_continuous_scale='Blues')
            fig.update_xaxes(side="bottom")
            fig.update_traces(
                textfont_size=16,  # Set the text font size (default is typically around 12)
                # textfont_color="black",  # Set text color if needed, especially for visibility on light backgrounds
                textfont_family="Arial Black"
            )

            st.plotly_chart(fig,use_container_width=True)
            with st.expander('Click here to display the underlying data'):
                st.write(data_cm[selected_features])
        else:
            st.write("Please select at least two data series to display the correlation matrix.")

    


############################################ ML Modelling####################################################
    with tab4:
        
        
        with st.form(key='ML_Form'):
            st.markdown("<h4>Enter Ticker:</h4>",unsafe_allow_html=True)
            stock_name = st.text_input('', '')
            st.markdown("<h4>Select Duration:</h4>",unsafe_allow_html=True)
            user_date = st.selectbox('', ["5 Year", "7 Years", "10 Years", "Custom Date"])
        # If stock name is entered, show the date input
           
            if user_date == "Custom Date":
                start_date = st.date_input("Select Start Date", value=datetime.date.today() - datetime.timedelta(days=5*365), min_value=datetime.date(1900, 1, 1), max_value=datetime.date.today())
                end_date = datetime.date.today()
            else:
                start_date, end_date = get_start_end_dates(user_date)

            st.markdown("<h4>Select the prediction model:</h4>",unsafe_allow_html=True)
            model_selection = st.selectbox(
                "",
                options=["XGBoost", "LSTM", "FB-Prophet"])
            
            # Submit button should be the last element in the form
            submitted = st.form_submit_button("Submit")
            
            if submitted:
                if stock_name and user_date and model_selection:
                    #Down the stock data based on user selection
                    end_date = datetime.date.today()
                    symbol = stock_name
                    data = yf.download(symbol, start=start_date, end=end_date)

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
                    class ProgressCallback(Callback):
                        def __init__(self, st_progress_bar):
                            super(ProgressCallback, self).__init__()
                            self.st_progress_bar = st_progress_bar  # Streamlit progress bar object

                        def on_epoch_end(self, epoch, logs=None):
                            # Update the progress bar
                            progress = (epoch + 1) / self.params['epochs']
                            self.st_progress_bar.progress(progress)
                            
                    if model_selection == "LSTM":
                        st.markdown("<h3>Training the model and forecasting the price. Please wait ⌛</h3>",unsafe_allow_html=True)
                        st_progress_bar = st.progress(0)
                        # Features and Target
                        features = df_combined[['Prev Adj Close', 'Bank_Interest']]  # Including Previous Day's Adj Close and Interest
                        target = df_combined['Adj Close']

                        # Scaling the features and target
                        scaler_features = MinMaxScaler(feature_range=(0, 1))
                        scaled_features = scaler_features.fit_transform(features)
                        scaler_target = MinMaxScaler(feature_range=(0, 1))
                        scaled_target = scaler_target.fit_transform(target.values.reshape(-1, 1))

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

                        # history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)
                        history = model.fit(
                            X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1,
                            callbacks=[ProgressCallback(st_progress_bar)]  # Add the custom callback here
                        )
                        # Prediction
                        y_pred = model.predict(X_test)

                        # Inverse scaling for plotting and evaluation
                        y_test_inv = scaler_target.inverse_transform(y_test)
                        y_pred_inv = scaler_target.inverse_transform(y_pred)

                        ## Predict the Price for next 30 period                    
                        last_steps = scaled_features[-time_steps:]
                        last_steps = last_steps.reshape((1, time_steps, scaled_features.shape[1]))
                        predicted_future = []
                        current_step = last_steps
                        
                        periods = 7

                        for i in range(periods):  # Predict the next 30 days
                            next_step = model.predict(current_step)
                            predicted_future.append(next_step[0, 0])

                            # Construct the new timestep by combining the predicted value with a placeholder for the second feature
                            next_features = np.array([next_step[0, 0], current_step[0, -1, 1]]).reshape(1, 1, 2)
                            
                            # Append the new timestep to current_step for the next prediction
                            current_step = np.append(current_step[:, 1:, :], next_features, axis=1)

                        predicted_future = np.array(predicted_future).reshape(-1, 1)
                        predicted_future = scaler_target.inverse_transform(predicted_future)

                        last_date = df_combined.index[-1]
                        date_range = pd.date_range(start=last_date, periods=periods+1, inclusive='right')
                        predicted_df = pd.DataFrame(data=predicted_future.flatten(), index=date_range, columns=['Predicted Adj Close'])

                        ## Calculate the error metrics
                        mae = mean_absolute_error(y_test_inv, y_pred_inv)
                        mse = mean_squared_error(y_test_inv, y_pred_inv)
                        rmse = np.sqrt(mse)
                        mape = mean_absolute_percentage_error(y_test_inv, y_pred_inv)
                        r_square = r2_score(y_test_inv, y_pred_inv)

                        ## Plot the Actual vs Predicted Price charts along with the Future Predicted values.

                        # Extract the actual dates for historical data
                        historical_dates = df_combined.index[split+time_steps:]  # Adjust this based on how split and time_steps are defined
                        historical_dates = historical_dates[:len(y_test_inv)]  # Ensure dates align with the test data

                        # Future predicted dates from predicted_df
                        future_dates = predicted_df.index

                        # Combine historical and future dates for plotting
                        total_dates = historical_dates.union(future_dates)

                        # Convert the date to a format suitable for plotting
                        total_dates_str = total_dates.strftime('%Y-%m-%d')

                        # Create traces for historical actual and predicted values
                        chart1 = go.Scatter(
                            x = historical_dates,
                            y = y_test_inv.flatten(),
                            mode = 'lines',
                            name = 'Actual'
                        )
                        chart2 = go.Scatter(
                            x = historical_dates,
                            y = y_pred_inv.flatten(),
                            mode = 'lines',
                            name = 'Predicted'
                        )

                        # Create trace for future predictions
                        chart3 = go.Scatter(
                            x = future_dates,
                            y = predicted_df['Predicted Adj Close'].values,
                            mode = 'lines',
                            name = 'Future Predicted'
                        )

                        # Layout
                        layout = go.Layout(
                            title = {
                                'text': f'Actual vs Forecasted Adj. Close Price of {symbol}',
                                'x': 0.5,  # Center title
                                'xanchor': 'center'
                            },
                            xaxis_title = 'Date',
                            yaxis_title = 'Adjusted Close Price',
                            width=1000,
                            height=600
                        )

                        # Figure
                        fig_lstm = go.Figure(data=[chart1, chart2, chart3], layout=layout)

                        # Show the plot
                        st.plotly_chart(fig_lstm,use_container_width=True)
                        st.markdown("---")
                    
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
                            st.markdown("<h3>Training the model and forecasting the price. Please wait ⌛</h3>",unsafe_allow_html=True)
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
                        st.plotly_chart(fig_xgb_f,use_container_width=True)
                        msg = f"The Adjusted Close Price on {df_combined.tail(1).index.date[0].strftime('%Y-%m-%d')} is $ {df_final.tail(1)['Target'].values[0]:.2f}  and the model's prediction was $ {df_final.tail(1)['Predicted'].values[0]:.2f}"
                        
                        st.markdown(f"""
                                    <div style='font-size: 20px;'>
                                        <strong>{msg}</strong>
                                    </div>
                                    """, unsafe_allow_html=True)
                        st.markdown("---")

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
                        st.plotly_chart(fig_fb,use_container_width=True)

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
                        st.plotly_chart(fig_trend,use_container_width=True)

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
                        st.plotly_chart(fig_yearly,use_container_width=True)

################## The below code is for the Sentiment Analysis #####################################################
    # with tab5:
    #     stock_sentiment_ticker=st.text_input('Name of Stock',value="")
        
    #     if stock_sentiment_ticker:
            
    #         sn=StockNews(stock_sentiment_ticker,save_news=False)
    #         df_news= sn.read_rss()
    #         if len(df_news) != 0:
    #             st.subheader(f'News for the ticker: {stock_sentiment_ticker}')
    #             df_news['published'] = pd.to_datetime(df_news['published']).dt.strftime("%Y-%m-%d %H:%M")
    #             for i in range(50):
    #                 st.write(f"<b>{df_news['title'][i]} - </b><small>{df_news['published'][i]}</small>",unsafe_allow_html=True)
    #                 st.caption(df_news['summary'][i])
    #                 title_sentiment= df_news['sentiment_title'][i]
    #                 news_sentiment= df_news['sentiment_summary'][i]
    #                 st.write(f'<b>Title Sentiment: {color_coded_sentiment(title_sentiment)} | \
    #                         News Sentiment: {color_coded_sentiment(news_sentiment)}<b>',unsafe_allow_html=True)
    #                 # st.write(f'News Sentiment:', color_coded_sentiment(news_sentiment),unsafe_allow_html=True)
    #                 st.markdown('---')
    #         else:
    #             st.subheader(f"No Articles found for the ticker: {stock_sentiment_ticker}, Please try with other value.")
    with tab5:
            stock_sentiment_ticker=st.text_input('Name of Stock',value="")
           
            if stock_sentiment_ticker:
           
                sn=StockNews(stock_sentiment_ticker,save_news=False)
                df_news= sn.read_rss()
                
                # df_news = sn.top_news(tickers=ticker)
                # st.write(df_news)
                if len(df_news) != 0:
                    st.subheader(f'News for the ticker: {stock_sentiment_ticker}')
                    df_news['published'] = pd.to_datetime(df_news['published']).dt.strftime("%Y-%m-%d %H:%M")
                    # import pdb;pdb.set_trace()
                    negative = df_news[df_news['sentiment_summary'] < 0]['sentiment_summary']
                    positive = df_news[df_news['sentiment_summary'] > 0]['sentiment_summary']
                    neutral = df_news[df_news['sentiment_summary'] == 0]['sentiment_summary']
                    counts = [negative.count(), positive.count(), neutral.count()]
                    # Create a figure for the trend component
                    fig_trend = go.Figure()
                    fig_trend.add_trace(go.Bar(
                        x=['Negative', 'Positive', 'Neutral'],
                        y=counts,
                        marker_color=['red', 'green', 'white'],  # Color coding the bars
                        name='Sentiment Count',
                        text=counts,
                        textposition='auto'
                    ))
 
                    # Update layout using the given structure
                    fig_trend.update_layout(
                        title={
                            'text': 'Sentiment Summary',
                            'x': 0.5,  # Center align the title
                            'xanchor': 'center'
                        },
                        xaxis_title='Sentiment',
                        yaxis_title='Count',
                        width=500,
                        height=500
                    )
 
                    # Show the plot, assuming using Streamlit
                    st.plotly_chart(fig_trend,use_container_width=True)
                    
                    with st.expander("News data"):
                        st.write(df_news)
 
 
                    # for i in range(50):
                    #     st.write(f"<b>{df_news['title'][i]} - </b><small>{df_news['published'][i]}</small>",unsafe_allow_html=True)
                    #     st.caption(df_news['summary'][i])
                    #     title_sentiment= df_news['sentiment_title'][i]
                    #     news_sentiment= df_news['sentiment_summary'][i]
                    #     st.write(f'<b>Title Sentiment: {color_coded_sentiment(title_sentiment)} | \
                    #             News Sentiment: {color_coded_sentiment(news_sentiment)}<b>',unsafe_allow_html=True)
                    #     st.write(f'News Sentiment:', color_coded_sentiment(news_sentiment),unsafe_allow_html=True)
                    #     st.markdown('---')
                else:
                    st.subheader(f"No Articles found for the ticker: {stock_sentiment_ticker}, Please try with other value.")


if __name__ == '__main__':
    main()
