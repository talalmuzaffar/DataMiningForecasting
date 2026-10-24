"""Matplotlib figures comparing a forecast with the actual values."""
import matplotlib.pyplot as plt


def plot_forecast_vs_actual(train_data, train_actual, test_data, test_actual, df, forecast_type):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_data.index, train_actual, label='Train')
    ax.plot(test_data.index, test_actual, label='Test')
    ax.plot(df['index'], df['Forecast'], label='Forecast')
    ax.set_xlabel('Date')
    ax.set_ylabel('Total Load Actual')
    ax.set_title(f'Total Load Actual Forecast vs Actual ({forecast_type})')
    ax.legend()
    return fig


def plot_column_forecast(test_data, df, column, forecast_type):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(test_data.index[:10], test_data[column][:10], label='Actual')
    ax.plot(test_data.index[:10], df[column], label='Forecast')
    ax.set_xlabel('Date')
    ax.set_ylabel(column)
    ax.set_title(f'{column} Forecast vs Actual ({forecast_type})')
    ax.legend()
    return fig


def plot_difference(test_data, df, forecast_type):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(test_data.index[:10], df['total load actual'][:10] - df['Forecast'], label='Difference')
    ax.set_xlabel('Date')
    ax.set_ylabel('Difference')
    ax.set_title(f'Difference Between Forecast and Actual Values ({forecast_type})')
    ax.axhline(0, color='black', linestyle='--')
    ax.legend()
    return fig
