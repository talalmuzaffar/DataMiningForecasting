"""Loading the saved models and making forecasts with them."""
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf


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


# Function to make predictions
def make_forecast(model, input_data, forecast_type, test_data):
    if forecast_type == 'Prophet':
        future = pd.DataFrame({'ds': test_data.index[:len(input_data)]})
        forecast = model.predict(future)['yhat'].values
    elif forecast_type == 'SVR':
        input_data = np.array(input_data).reshape(-1, 1)
        forecast = model.predict(input_data)
    elif forecast_type == 'ETS':
        forecast = model.forecast(len(input_data))
    elif forecast_type == 'LSTM':
        input_data = np.array(input_data).reshape((1, len(input_data), 1))
        forecast = model.predict(input_data)
        forecast = forecast.flatten()
    else:
        forecast = []
    return forecast
