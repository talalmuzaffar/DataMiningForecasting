"""Streamlit page showing a model's forecast, accuracy metrics and charts."""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_squared_error, mean_absolute_error

from data_mining_forecasting.charts import plot_column_forecast, plot_difference, plot_forecast_vs_actual
from data_mining_forecasting.models import make_forecast


# Main Content
def display_forecast_page(forecast_type, model, test_data, test_actual, train_data, train_actual):
    if model is None:
        st.error(f"{forecast_type} model is not available")
        return

    st.title(f'Load Forecast ({forecast_type})')
    total_load_forecast = st.text_area('Enter Total Load Forecast (comma-separated)', '')

    if st.button(f'Get {forecast_type} Forecast'):
        total_load_forecast = [float(i) for i in total_load_forecast.split(',')]
        if len(total_load_forecast) < 10:
            st.error("Please enter at least 10 forecasts.")
        else:
            forecast = make_forecast(model, total_load_forecast[:10], forecast_type, test_data)
            mse = mean_squared_error(test_actual[:len(forecast)], forecast)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(test_actual[:len(forecast)], forecast)

            st.subheader('Accuracy Metrics:')
            st.write(f'Mean Squared Error (MSE): {mse}')
            st.write(f'Root Mean Squared Error (RMSE): {rmse}')
            st.write(f'Mean Absolute Error (MAE): {mae}')

            forecast_df = pd.DataFrame({'Forecast': forecast}, index=test_data.index[:10])
            df = pd.concat([test_data[:10].reset_index(), forecast_df.reset_index(drop=True)], axis=1)

            st.subheader('Forecast Data:')
            st.dataframe(df)

            fig = plot_forecast_vs_actual(train_data, train_actual, test_data, test_actual, df, forecast_type)
            st.pyplot(fig)

            for column in test_data.columns[:-1]:
                fig = plot_column_forecast(test_data, df, column, forecast_type)
                st.pyplot(fig)

            fig = plot_difference(test_data, df, forecast_type)
            st.pyplot(fig)
