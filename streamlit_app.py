import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Suppress warnings
st.set_option('deprecation.showPyplotGlobalUse', False)

# Load test and train data
test_data = pd.read_csv('test_data.csv', index_col=0, parse_dates=True)
test_actual = test_data['total load actual']
train_data = pd.read_csv('train_data.csv', index_col=0, parse_dates=True)
train_actual = train_data['total load actual']

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["SARIMAX Forecast", "ARIMA Forecast", "ETS Forecast","Prophet Forecast","SVR Forecast","ANN Forecast","LSTM Forecast","Hybrid Forecast"])

# SARIMAX Forecast Page
if page == "SARIMAX Forecast":
    st.title('Total Load Actual Forecast (SARIMAX)')
    
    # User input
    total_load_forecast = st.text_area('Total Load Forecast (comma-separated)', '')

    if st.button('Get SARIMAX Forecast'):
        # Convert user input to list of floats
        total_load_forecast = [float(i) for i in total_load_forecast.split(',')]
        
        # Ensure we have at least 10 forecasts
        if len(total_load_forecast) < 10:
            st.error("Please enter at least 10 forecasts.")
        else:
            # Make request to Flask backend
            data = {'total load forecast': total_load_forecast[:10]}
            response = requests.post('http://localhost:5000/forecast_sarimax', json=data)
            forecast = response.json()['forecast']
            
            # Calculate MSE
            mse = mean_squared_error(test_actual[:len(forecast)], forecast)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(test_actual[:len(forecast)], forecast)
            st.write('Accuracy Metrics:')
            st.write(f'Mean Squared Error (MSE): {mse}')
            st.write(f'Root Mean Squared Error (RMSE): {rmse}')
            st.write(f'Mean Absolute Error (MAE): {mae}')
            
            # Format forecast data into DataFrame
            forecast_df = pd.DataFrame({'Forecast': forecast}, index=test_data.index[:10])
            
            # Visualize forecast and actual data
            df = pd.concat([test_data[:10].reset_index(), forecast_df.reset_index(drop=True)], axis=1)
            
            st.write('Forecast Data:')
            st.write(df)
            
            plt.figure(figsize=(10, 6))
            plt.plot(train_data.index, train_actual, label='Train')
            plt.plot(test_data.index, test_actual, label='Test')
            plt.plot(df['index'], df['Forecast'], label='Forecast')
            plt.xlabel('Date')
            plt.ylabel('Total Load Actual')
            plt.title('Total Load Actual Forecast vs Actual (SARIMAX)')
            plt.legend()
            st.pyplot()

            for column in test_data.columns[:-1]:
                plt.figure(figsize=(10, 6))
                plt.plot(test_data.index[:10], test_data[column][:10], label='Actual')
                plt.plot(test_data.index[:10], df[column], label='Forecast')
                plt.xlabel('Date')
                plt.ylabel(column)
                plt.title(f'{column} Forecast vs Actual (SARIMAX)')
                plt.legend()
                st.pyplot()

            # Plot difference between forecast and actual values
            plt.figure(figsize=(10, 6))
            plt.plot(test_data.index[:10], df['total load actual'][:10] - df['Forecast'], label='Difference')
            plt.xlabel('Date')
            plt.ylabel('Difference')
            plt.title('Difference Between Forecast and Actual Values (SARIMAX)')
            plt.axhline(0, color='black', linestyle='--')
            plt.legend()
            st.pyplot()

            plt.subplot(1, 2, 2)
            residuals = test_actual[:len(forecast)] - forecast
            plt.scatter(forecast, residuals)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Forecast')
            plt.ylabel('Residuals')
            plt.title('Residuals vs Forecast')
                    
            st.pyplot()


# ARIMA Forecast Page
elif page == "ARIMA Forecast":
    st.title('Total Load Actual Forecast (ARIMA)')
    
    # User input
    total_load_forecast = st.text_area('Total Load Forecast (comma-separated)', '')

    if st.button('Get ARIMA Forecast'):
        # Convert user input to list of floats
        total_load_forecast = [float(i) for i in total_load_forecast.split(',')]
        
        # Ensure we have at least 10 forecasts
        if len(total_load_forecast) < 10:
            st.error("Please enter at least 10 forecasts.")
        else:
            # Make request to Flask backend
            data = {'total load forecast': total_load_forecast[:10]}
            response = requests.post('http://localhost:5000/forecast_arima', json=data)
            forecast = response.json()['forecast']
            
            # Calculate MSE
            mse = mean_squared_error(test_actual[:len(forecast)], forecast)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(test_actual[:len(forecast)], forecast)
            st.write('Accuracy Metrics:')
            st.write(f'Mean Squared Error (MSE): {mse}')
            st.write(f'Root Mean Squared Error (RMSE): {rmse}')
            st.write(f'Mean Absolute Error (MAE): {mae}')
            
            # Format forecast data into DataFrame
            forecast_df = pd.DataFrame({'Forecast': forecast}, index=test_data.index[:10])
            
            # Visualize forecast and actual data
            df = pd.concat([test_data[:10].reset_index(), forecast_df.reset_index(drop=True)], axis=1)
            
            st.write('Forecast Data:')
            st.write(df)
            
            plt.figure(figsize=(10, 6))
            plt.plot(train_data.index, train_actual, label='Train')
            plt.plot(test_data.index, test_actual, label='Test')
            plt.plot(df['index'], df['Forecast'], label='Forecast')
            plt.xlabel('Date')
            plt.ylabel('Total Load Actual')
            plt.title('Total Load Actual Forecast vs Actual (ARIMA)')
            plt.legend()
            st.pyplot()


            for column in test_data.columns[:-1]:
                plt.figure(figsize=(10, 6))
                plt.plot(test_data.index[:10], test_data[column][:10], label='Actual')
                plt.plot(test_data.index[:10], df[column], label='Forecast')
                plt.xlabel('Date')
                plt.ylabel(column)
                plt.title(f'{column} Forecast vs Actual (SARIMAX)')
                plt.legend()
                st.pyplot()

            # Plot difference between forecast and actual values
            plt.figure(figsize=(10, 6))
            plt.plot(test_data.index[:10], df['total load actual'][:10] - df['Forecast'], label='Difference')
            plt.xlabel('Date')
            plt.ylabel('Difference')
            plt.title('Difference Between Forecast and Actual Values (SARIMAX)')
            plt.axhline(0, color='black', linestyle='--')
            plt.legend()
            st.pyplot()

            plt.subplot(1, 2, 2)
            residuals = test_actual[:len(forecast)] - forecast
            plt.scatter(forecast, residuals)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Forecast')
            plt.ylabel('Residuals')
            plt.title('Residuals vs Forecast')
                    
            st.pyplot()


# ETS Forecast Page
elif page == "ETS Forecast":
    st.title('Total Load Actual Forecast (ETS)')
    
    # User input
    total_load_forecast = st.text_area('Total Load Forecast (comma-separated)', '')

    if st.button('Get ETS Forecast'):
        # Convert user input to list of floats
        total_load_forecast = [float(i) for i in total_load_forecast.split(',')]
        
        # Ensure we have at least 10 forecasts
        if len(total_load_forecast) < 10:
            st.error("Please enter at least 10 forecasts.")
        else:
            # Make request to Flask backend
            data = {'total load forecast': total_load_forecast[:10]}
            response = requests.post('http://localhost:5000/forecast_ets', json=data)
            forecast = response.json()['forecast']
            
            # Calculate MSE
            mse = mean_squared_error(test_actual[:len(forecast)], forecast)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(test_actual[:len(forecast)], forecast)
            st.write('Accuracy Metrics:')
            st.write(f'Mean Squared Error (MSE): {mse}')
            st.write(f'Root Mean Squared Error (RMSE): {rmse}')
            st.write(f'Mean Absolute Error (MAE): {mae}')
            
            # Format forecast data into DataFrame
            forecast_df = pd.DataFrame({'Forecast': forecast}, index=test_data.index[:10])
            
            # Visualize forecast and actual data
            df = pd.concat([test_data[:10].reset_index(), forecast_df.reset_index(drop=True)], axis=1)
            
            st.write('Forecast Data:')
            st.write(df)
            
            plt.figure(figsize=(10, 6))
            plt.plot(train_data.index, train_actual, label='Train')
            plt.plot(test_data.index, test_actual, label='Test')
            plt.plot(df['index'], df['Forecast'], label='Forecast')
            plt.xlabel('Date')
            plt.ylabel('Total Load Actual')
            plt.title('Total Load Actual Forecast vs Actual (ETS)')
            plt.legend()
            st.pyplot()


            for column in test_data.columns[:-1]:
                plt.figure(figsize=(10, 6))
                plt.plot(test_data.index[:10], test_data[column][:10], label='Actual')
                plt.plot(test_data.index[:10], df[column], label='Forecast')
                plt.xlabel('Date')
                plt.ylabel(column)
                plt.title(f'{column} Forecast vs Actual (SARIMAX)')
                plt.legend()
                st.pyplot()

            # Plot difference between forecast and actual values
            plt.figure(figsize=(10, 6))
            plt.plot(test_data.index[:10], df['total load actual'][:10] - df['Forecast'], label='Difference')
            plt.xlabel('Date')
            plt.ylabel('Difference')
            plt.title('Difference Between Forecast and Actual Values (SARIMAX)')
            plt.axhline(0, color='black', linestyle='--')
            plt.legend()
            st.pyplot()

            plt.subplot(1, 2, 2)
            residuals = test_actual[:len(forecast)] - forecast
            plt.scatter(forecast, residuals)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Forecast')
            plt.ylabel('Residuals')
            plt.title('Residuals vs Forecast')
                    
            st.pyplot()


elif page == "Prophet Forecast":
    st.title('Total Load Actual Forecast (Prophet)')
    
    # User input
    total_load_forecast = st.text_area('Total Load Forecast (comma-separated)', '')

    if st.button('Get Prophet Forecast'):
        # Convert user input to list of floats
        total_load_forecast = [float(i) for i in total_load_forecast.split(',')]
        
        # Ensure we have at least 10 forecasts
        if len(total_load_forecast) < 10:
            st.error("Please enter at least 10 forecasts.")
        else:
            # Make request to Flask backend
            data = {'total load forecast': total_load_forecast[:10]}
            response = requests.post('http://localhost:5000/forecast_prophet', json=data)
            forecast = response.json()['forecast']
            
            mse = mean_squared_error(test_actual[:len(forecast)], forecast)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(test_actual[:len(forecast)], forecast)
            st.write('Accuracy Metrics:')
            st.write(f'Mean Squared Error (MSE): {mse}')
            st.write(f'Root Mean Squared Error (RMSE): {rmse}')
            st.write(f'Mean Absolute Error (MAE): {mae}')
            
            # Format forecast data into DataFrame
            forecast_df = pd.DataFrame({'Forecast': forecast}, index=test_data.index[:10])
            
            # Visualize forecast and actual data
            df = pd.concat([test_data[:10].reset_index(), forecast_df.reset_index(drop=True)], axis=1)
            
            st.write('Forecast Data:')
            st.write(df)
            
            # Plot actual vs forecast
            plt.figure(figsize=(10, 6))
            plt.plot(train_data.index, train_actual, label='Train')
            plt.plot(test_data.index, test_actual, label='Test')
            plt.plot(df['index'], df['Forecast'], label='Forecast')
            plt.xlabel('Date')
            plt.ylabel('Total Load Actual')
            plt.title('Total Load Actual Forecast vs Actual (Prophet)')
            plt.legend()
            st.pyplot()

            for column in test_data.columns[:-1]:
                plt.figure(figsize=(10, 6))
                plt.plot(test_data.index[:10], test_data[column][:10], label='Actual')
                plt.plot(test_data.index[:10], df[column], label='Forecast')
                plt.xlabel('Date')
                plt.ylabel(column)
                plt.title(f'{column} Forecast vs Actual (SARIMAX)')
                plt.legend()
                st.pyplot()

            # Plot difference between forecast and actual values
            plt.figure(figsize=(10, 6))
            plt.plot(test_data.index[:10], df['total load actual'][:10] - df['Forecast'], label='Difference')
            plt.xlabel('Date')
            plt.ylabel('Difference')
            plt.title('Difference Between Forecast and Actual Values (SARIMAX)')
            plt.axhline(0, color='black', linestyle='--')
            plt.legend()
            st.pyplot()

            plt.subplot(1, 2, 2)
            residuals = test_actual[:len(forecast)] - forecast
            plt.scatter(forecast, residuals)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Forecast')
            plt.ylabel('Residuals')
            plt.title('Residuals vs Forecast')
                    
            st.pyplot()


elif page == "SVR Forecast":

    st.title('Total Load Actual Forecast (SVR)')
    
    # User input
    total_load_forecast = st.text_area('Total Load Forecast (comma-separated)', '')

    if st.button('Get SVR Forecast'):
         total_load_forecast = [float(i) for i in total_load_forecast.split(',')]
    
    # Ensure we have at least 10 forecasts
         if len(total_load_forecast) < 10:
            st.error("Please enter at least 10 forecasts.")
         else:
            # Make request to Flask backend
            data = {'total load forecast': total_load_forecast[:10]}
            response = requests.post('http://localhost:5000/forecast_svr', json=data)
            forecast = response.json()['forecast']
            
            mse = mean_squared_error(test_actual[:len(forecast)], forecast)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(test_actual[:len(forecast)], forecast)
            st.write('Accuracy Metrics:')
            st.write(f'Mean Squared Error (MSE): {mse}')
            st.write(f'Root Mean Squared Error (RMSE): {rmse}')
            st.write(f'Mean Absolute Error (MAE): {mae}')
            
            # Visualize forecast and actual data
            df = pd.concat([test_data[:10].reset_index(), pd.DataFrame({'Forecast': forecast})], axis=1)
            
            st.write('Forecast Data:')
            st.write(df)
            
            # Plot actual vs forecast
            plt.figure(figsize=(10, 6))
            plt.plot(train_data.index, train_actual, label='Train')
            plt.plot(test_data.index, test_actual, label='Test')
            plt.plot(df['index'], df['Forecast'], label='Forecast')
            plt.xlabel('Date')
            plt.ylabel('Total Load Actual')
            plt.title('Total Load Actual Forecast vs Actual (SVR)')
            plt.legend()
            st.pyplot()

            for column in test_data.columns[:-1]:
                plt.figure(figsize=(10, 6))
                plt.plot(test_data.index[:10], test_data[column][:10], label='Actual')
                plt.plot(test_data.index[:10], df[column], label='Forecast')
                plt.xlabel('Date')
                plt.ylabel(column)
                plt.title(f'{column} Forecast vs Actual (SARIMAX)')
                plt.legend()
                st.pyplot()

            # Plot difference between forecast and actual values
            plt.figure(figsize=(10, 6))
            plt.plot(test_data.index[:10], df['total load actual'][:10] - df['Forecast'], label='Difference')
            plt.xlabel('Date')
            plt.ylabel('Difference')
            plt.title('Difference Between Forecast and Actual Values (SARIMAX)')
            plt.axhline(0, color='black', linestyle='--')
            plt.legend()
            st.pyplot()

            plt.subplot(1, 2, 2)
            residuals = test_actual[:len(forecast)] - forecast
            plt.scatter(forecast, residuals)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Forecast')
            plt.ylabel('Residuals')
            plt.title('Residuals vs Forecast')
                    
            st.pyplot()

elif page == "ANN Forecast":

    st.title('Total Load Actual Forecast (SVR)')
    
    # User input
    total_load_forecast = st.text_area('Total Load Forecast (comma-separated)', '')


    if st.button('Get ANN Forecast'):
        # Convert user input to list of floats
        total_load_forecast = [float(i) for i in total_load_forecast.split(',')]
        
        # Ensure we have at least 10 forecasts
        if len(total_load_forecast) < 10:
            st.error("Please enter at least 10 forecasts.")
        else:
            # Make request to Flask backend
            data = {'total load forecast': total_load_forecast[:10]}
            response = requests.post('http://localhost:5000/forecast_ann', json=data)
            forecast = response.json()['forecast']

            mse = mean_squared_error(test_actual[:len(forecast)], forecast)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(test_actual[:len(forecast)], forecast)
            st.write('Accuracy Metrics:')
            st.write(f'Mean Squared Error (MSE): {mse}')
            st.write(f'Root Mean Squared Error (RMSE): {rmse}')
            st.write(f'Mean Absolute Error (MAE): {mae}')
            
            # Visualize forecast and actual data
            df = pd.concat([test_data[:10].reset_index(), pd.DataFrame({'Forecast': forecast})], axis=1)
            
            st.write('Forecast Data:')
            st.write(df)
            
            plt.figure(figsize=(10, 6))
            plt.plot(train_data.index, train_actual, label='Train')
            plt.plot(test_data.index, test_actual, label='Test')

            st.write('Forecast Values:', df['Forecast'])


            for column in test_data.columns[:-1]:
                plt.figure(figsize=(10, 6))
                plt.plot(test_data.index[:10], test_data[column][:10], label='Actual')
                plt.plot(test_data.index[:10], df[column], label='Forecast')
                plt.xlabel('Date')
                plt.ylabel(column)
                plt.title(f'{column} Forecast vs Actual (SARIMAX)')
                plt.legend()
                st.pyplot()

            plt.subplot(1, 2, 2)
            residuals = test_actual[:len(forecast)] - forecast
            plt.scatter(forecast, residuals)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Forecast')
            plt.ylabel('Residuals')
            plt.title('Residuals vs Forecast')
                    
            st.pyplot()

elif page == "LSTM Forecast":

    st.title('Total Load Actual Forecast (LSTM)')
    
    # User input
    total_load_forecast = st.text_area('Total Load Forecast (comma-separated)', '')


    if st.button('Get LSTM Forecast'):
        # Convert user input to list of floats
        total_load_forecast = [float(i) for i in total_load_forecast.split(',')]
        
        # Ensure we have at least 10 forecasts
        if len(total_load_forecast) < 10:
            st.error("Please enter at least 10 forecasts.")
        else:
            # Make request to Flask backend
            data = {'total load forecast': total_load_forecast[:10]}
            response = requests.post('http://localhost:5000/forecast_lstm', json=data)
            forecast = response.json()['forecast']

            mse = mean_squared_error(test_actual[:len(forecast)], forecast)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(test_actual[:len(forecast)], forecast)
            st.write('Accuracy Metrics:')
            st.write(f'Mean Squared Error (MSE): {mse}')
            st.write(f'Root Mean Squared Error (RMSE): {rmse}')
            st.write(f'Mean Absolute Error (MAE): {mae}')
            # Visualize forecast and actual data
            df = pd.concat([test_data[:10].reset_index(), pd.DataFrame({'Forecast': forecast})], axis=1)
            
            st.write('Forecast Data:')
            st.write(df)
            
            plt.figure(figsize=(10, 6))
            plt.plot(train_data.index, train_actual, label='Train')
            plt.plot(test_data.index, test_actual, label='Test')

            st.write('Forecast Values:', df['Forecast'])


            for column in test_data.columns[:-1]:
                plt.figure(figsize=(10, 6))
                plt.plot(test_data.index[:10], test_data[column][:10], label='Actual')
                plt.plot(test_data.index[:10], df[column], label='Forecast')
                plt.xlabel('Date')
                plt.ylabel(column)
                plt.title(f'{column} Forecast vs Actual (SARIMAX)')
                plt.legend()
                st.pyplot()

            plt.subplot(1, 2, 2)
            residuals = test_actual[:len(forecast)] - forecast
            plt.scatter(forecast, residuals)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Forecast')
            plt.ylabel('Residuals')
            plt.title('Residuals vs Forecast')
                    
            st.pyplot()


elif page == "Hybrid Forecast":

    st.title('Total Load Actual Forecast (Hybrid)')
    
    # User input
    total_load_forecast = st.text_area('Total Load Forecast (comma-separated)', '')


    if st.button('Get Hybrid Forecast'):
        # Convert user input to list of floats
        total_load_forecast = [float(i) for i in total_load_forecast.split(',')]
        
        # Ensure we have at least 10 forecasts
        if len(total_load_forecast) < 10:
            st.error("Please enter at least 10 forecasts.")
        else:
            # Make request to Flask backend
            data = {'total load forecast': total_load_forecast[:10]}
            response = requests.post('http://localhost:5000/forecast_hybrid', json=data)
            forecast = response.json()['forecast']

            mse = mean_squared_error(test_actual[:len(forecast)], forecast)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(test_actual[:len(forecast)], forecast)
            st.write('Accuracy Metrics:')
            st.write(f'Mean Squared Error (MSE): {mse}')
            st.write(f'Root Mean Squared Error (RMSE): {rmse}')
            st.write(f'Mean Absolute Error (MAE): {mae}')
            # Visualize forecast and actual data
            df = pd.concat([test_data[:10].reset_index(), pd.DataFrame({'Forecast': forecast})], axis=1)
            
            st.write('Forecast Data:')
            st.write(df)
            
            plt.figure(figsize=(10, 6))
            plt.plot(train_data.index, train_actual, label='Train')
            plt.plot(test_data.index, test_actual, label='Test')

            st.write('Forecast Values:', df['Forecast'])


            for column in test_data.columns[:-1]:
                plt.figure(figsize=(10, 6))
                plt.plot(test_data.index[:10], test_data[column][:10], label='Actual')
                plt.plot(test_data.index[:10], df[column], label='Forecast')
                plt.xlabel('Date')
                plt.ylabel(column)
                plt.title(f'{column} Forecast vs Actual (SARIMAX)')
                plt.legend()
                st.pyplot()

            plt.subplot(1, 2, 2)
            residuals = test_actual[:len(forecast)] - forecast
            plt.scatter(forecast, residuals)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Forecast')
            plt.ylabel('Residuals')
            plt.title('Residuals vs Forecast')
                    
            st.pyplot()
