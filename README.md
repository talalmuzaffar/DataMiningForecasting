# Energy Load Forecasting System 📊

A time series forecasting system that implements multiple machine learning and deep learning models for prediction. Built with Streamlit for an interactive user interface.

## 🌟 Features

- **Multiple Model Support**: 
  - Long Short-Term Memory (LSTM) Networks
  - Artificial Neural Networks (ANN)
  - Support Vector Regression (SVR)
  - Facebook Prophet
  - Exponential Time Series (ETS)
- **Interactive UI**: Built with Streamlit
- **Data Visualization**: Time series plots and analysis

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/talalmuzaffar/DataMiningForecasting.git
   cd DataMiningForecasting
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit application:
   ```bash
   streamlit run streamlit_app.py
   ```

## 📁 Project Structure

```
DataMiningForecasting/
├── streamlit_app.py            # Main Streamlit interface
├── app.py                      # Flask REST API serving the models (python app.py)
├── data_mining_forecasting/
│   ├── __init__.py
│   ├── config.py               # Paths to the datasets and model files
│   ├── data.py                 # Loads the train/test datasets
│   ├── models.py               # Loads the saved models and makes forecasts
│   ├── charts.py               # Forecast vs actual plots
│   ├── ui.py                   # Streamlit forecast page
│   └── api.py                  # Flask routes used by app.py
├── data/
│   ├── train_data.csv          # Training dataset
│   └── test_data.csv           # Testing dataset
├── models/
│   ├── lstm_model.h5           # LSTM model file
│   ├── ann_model.h5            # ANN model file
│   ├── svr_model.pkl           # SVR model file
│   ├── prophet_model.pkl       # Prophet model file
│   └── ets_model.pkl           # ETS model file
└── requirements.txt
```

## 👥 Authors

- Talal Muzaffar - [GitHub](https://github.com/talalmuzaffar)

## 📞 Support

For issues and questions, please [open an issue](https://github.com/talalmuzaffar/DataMiningForecasting/issues) on GitHub.
