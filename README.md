# Energy Load Forecasting System 📊⚡

A comprehensive energy load forecasting system that utilizes multiple machine learning and deep learning models to predict energy consumption patterns. This project implements various forecasting techniques including LSTM, ANN, SVR, Prophet, and ETS models through a user-friendly Streamlit interface.

## 🌟 Features

- **Multiple Model Support**: 
  - Long Short-Term Memory (LSTM) Networks
  - Artificial Neural Networks (ANN)
  - Support Vector Regression (SVR)
  - Facebook Prophet
  - Exponential Time Series (ETS)
- **Interactive UI**: Built with Streamlit for easy model selection and visualization
- **Data Preprocessing**: Automated data preparation and feature engineering
- **Model Comparison**: Compare performance across different forecasting models
- **Visualization**: Rich set of charts and graphs for result analysis
- **Time Series Analysis**: Specialized tools for temporal data analysis

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

## 💡 Usage

1. Launch the Streamlit application
2. Upload your time series data (or use provided test data)
3. Select the desired forecasting model(s)
4. Configure model parameters if needed
5. View and compare forecasting results
6. Export predictions and visualizations

## 📊 Models Included

### LSTM (Long Short-Term Memory)
- Specialized deep learning model for sequence prediction
- Handles long-term dependencies in time series data
- Stored in `lstm_model.h5`

### ANN (Artificial Neural Network)
- Feed-forward neural network for regression
- Capable of capturing non-linear patterns
- Stored in `ann_model.h5`

### SVR (Support Vector Regression)
- Non-linear regression using kernel methods
- Robust to outliers
- Stored in `svr_model.pkl`

### Prophet
- Facebook's time series forecasting tool
- Handles seasonality and holiday effects
- Stored in `prophet_model.pkl`

### ETS (Error, Trend, Seasonality)
- Statistical model for time series decomposition
- Captures trend and seasonal components
- Stored in `ets_model.pkl`

## 📁 Project Structure

- `streamlit_app.py`: Main application interface
- `app.py`: Core application logic
- `train_data.csv`: Training dataset
- `test_data.csv`: Testing dataset
- Model files:
  - `lstm_model.h5`
  - `ann_model.h5`
  - `svr_model.pkl`
  - `prophet_model.pkl`
  - `ets_model.pkl`

## 🔧 Technical Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **ML/DL Libraries**: 
  - TensorFlow/Keras
  - scikit-learn
  - Facebook Prophet
  - statsmodels
- **Data Processing**: 
  - pandas
  - numpy
- **Visualization**: 
  - matplotlib
  - plotly

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 👥 Authors

- Talal Muzaffar - *Initial work* - [GitHub](https://github.com/talalmuzaffar)

## 🙏 Acknowledgments

- TensorFlow team for the deep learning framework
- Facebook Research for Prophet
- Streamlit team for the excellent web framework

## 📞 Support

If you encounter any issues or have questions, please [open an issue](https://github.com/talalmuzaffar/DataMiningForecasting/issues) on GitHub.
