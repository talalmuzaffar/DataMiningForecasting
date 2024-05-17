import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import tensorflow as tf
import os

# Suppress warnings
st.set_option('deprecation.showPyplotGlobalUse', False)

# Load test and train data
test_data = pd.read_csv('test_data.csv', index_col=0, parse_dates=True)
test_actual = test_data['total load actual']
train_data = pd.read_csv('train_data.csv', index_col=0, parse_dates=True)
train_actual = train_data['total load actual']

# Load models
def load_model(file_path):
    if os.path.exists(file_path):
        try:
            if file_path.endswith('.pkl'):
                return joblib.load(file_path)
            elif file_path.endswith('.h5'):
                return tf.keras.models.load_model(file_path)
        except Exception as e:
            st.error(f"Error loading model {file_path}: {e}")
    else:
        st.error(f"Model file {file_path} not found")
    return None

ets_model = load_model('ets_model.pkl')
prophet_model = load_model('prophet_model.pkl')
svr_model = load_model('svr_model.pkl')
ann_model = load_model('ann_model.h5')
lstm_model = load_model('lstm_model.h5')

# Function to make predictions
def make_forecast(model, input_data, forecast_type):
    if forecast_type in ['ETS', 'Prophet', 'SVR']:
        forecast = model.forecast(len(input_data))
    elif forecast_type in ['ANN', 'LSTM']:
        input_data = np.array(input_data).reshape((1, len(input_data), 1))
        forecast = model.predict(input_data)
        forecast = forecast.flatten()
    else:
        forecast = []
    return forecast

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["ETS Forecast", "Prophet Forecast", "SVR Forecast", "ANN Forecast", "LSTM Forecast"])

# Forecasting Pages
def display_forecast_page(forecast_type, model):
    if model is None:
        st.error(f"Model for {forecast_type} is not available")
        return

    st.title(f'Total Load Actual Forecast ({forecast_type})')
    
    # User input
    total_load_forecast = st.text_area('Total Load Forecast (comma-separated)', '')
    
    if st.button(f'Get {forecast_type} Forecast'):
        # Convert user input to list of floats
        total_load_forecast = [float(i) for i in total_load_forecast.split(',')]
        
        # Ensure we have at least 10 forecasts
        if len(total_load_forecast) < 10:
            st.error("Please enter at least 10 forecasts.")
        else:
            # Make forecast
            forecast = make_forecast(model, total_load_forecast[:10], forecast_type)
            
            # Calculate metrics
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
            plt.title(f'Total Load Actual Forecast vs Actual ({forecast_type})')
            plt.legend()
            st.pyplot()
            
            for column in test_data.columns[:-1]:
                plt.figure(figsize=(10, 6))
                plt.plot(test_data.index[:10], test_data[column][:10], label='Actual')
                plt.plot(test_data.index[:10], df[column], label='Forecast')
                plt.xlabel('Date')
                plt.ylabel(column)
                plt.title(f'{column} Forecast vs Actual ({forecast_type})')
                plt.legend()
                st.pyplot()
            
            plt.figure(figsize=(10, 6))
            plt.plot(test_data.index[:10], df['total load actual'][:10] - df['Forecast'], label='Difference')
            plt.xlabel('Date')
            plt.ylabel('Difference')
            plt.title(f'Difference Between Forecast and Actual Values ({forecast_type})')
            plt.axhline(0, color='black', linestyle='--')
            plt.legend()
            st.pyplot()

if page == "ETS Forecast":
    display_forecast_page("ETS", ets_model)
elif page == "Prophet Forecast":
    display_forecast_page("Prophet", prophet_model)
elif page == "SVR Forecast":
    display_forecast_page("SVR", svr_model)
elif page == "ANN Forecast":
    display_forecast_page("ANN", ann_model)
elif page == "LSTM Forecast":
    display_forecast_page("LSTM", lstm_model)
