import streamlit as st

from data_mining_forecasting.config import ETS_MODEL_PATH, LSTM_MODEL_PATH, PROPHET_MODEL_PATH, SVR_MODEL_PATH
from data_mining_forecasting.data import load_data
from data_mining_forecasting.models import load_model
from data_mining_forecasting.ui import display_forecast_page

# Load test and train data
test_data, test_actual, train_data, train_actual = load_data()

# Load models
ets_model = load_model(ETS_MODEL_PATH)
prophet_model = load_model(PROPHET_MODEL_PATH)
svr_model = load_model(SVR_MODEL_PATH)
lstm_model = load_model(LSTM_MODEL_PATH)

# Sidebar Navigation
st.sidebar.title("Forecasting Models")
page = st.sidebar.radio("Select Model", ["ETS Forecast", "Prophet Forecast", "SVR Forecast", "LSTM Forecast"])

if page == "ETS Forecast":
    display_forecast_page("ETS", ets_model, test_data, test_actual, train_data, train_actual)
elif page == "Prophet Forecast":
    display_forecast_page("Prophet", prophet_model, test_data, test_actual, train_data, train_actual)
elif page == "SVR Forecast":
    display_forecast_page("SVR", svr_model, test_data, test_actual, train_data, train_actual)
elif page == "LSTM Forecast":
    display_forecast_page("LSTM", lstm_model, test_data, test_actual, train_data, train_actual)
