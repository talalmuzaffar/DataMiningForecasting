# Energy Load Forecasting System 📊⚡

A comprehensive energy load forecasting system that leverages state-of-the-art machine learning and deep learning models to predict energy consumption patterns with high accuracy. This project implements an ensemble of advanced forecasting techniques through an intuitive Streamlit interface, making it accessible for both analysts and data scientists.

## 🌟 Features

- **Advanced Model Ensemble**: 
  - Long Short-Term Memory (LSTM) Networks for complex sequential patterns
  - Artificial Neural Networks (ANN) for non-linear relationships
  - Support Vector Regression (SVR) with kernel optimization
  - Facebook Prophet for robust trend decomposition
  - Exponential Time Series (ETS) for statistical forecasting
- **Interactive UI**: 
  - Built with Streamlit for intuitive model interaction
  - Real-time parameter tuning
  - Dynamic visualization updates
  - Model performance comparison dashboard
- **Intelligent Data Preprocessing**: 
  - Automated missing value imputation
  - Outlier detection and handling
  - Feature scaling and normalization
  - Temporal feature engineering
- **Advanced Model Comparison**: 
  - Cross-validation metrics (RMSE, MAE, MAPE)
  - Model performance visualization
  - Ensemble weighting options
  - Confidence interval estimation
- **Comprehensive Visualization**: 
  - Interactive time series plots
  - Forecast vs. actual comparisons
  - Error analysis dashboards
  - Component decomposition views
- **Time Series Analysis Tools**: 
  - Seasonality detection
  - Trend analysis
  - Autocorrelation analysis
  - Stationarity tests

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)
- 8GB RAM minimum (16GB recommended for large datasets)
- CUDA-compatible GPU (optional, for faster LSTM training)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/talalmuzaffar/DataMiningForecasting.git
   cd DataMiningForecasting
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit application:
   ```bash
   streamlit run streamlit_app.py
   ```

## 💡 Usage

1. Launch the Streamlit application
2. Data Input:
   - Upload your time series data in CSV format
   - Required columns: timestamp, target_variable
   - Optional: external features
   - Use provided test data for demo
3. Model Selection:
   - Choose single or multiple models
   - Configure model-specific parameters
   - Set training/validation split
4. Training Configuration:
   - Set forecast horizon
   - Configure cross-validation strategy
   - Adjust hyperparameters
5. Analysis and Results:
   - View model performance metrics
   - Analyze forecast visualizations
   - Compare model predictions
   - Export results and visualizations

## 📊 Models in Detail

### LSTM (Long Short-Term Memory)
- Architecture: Multi-layer LSTM with dropout
- Input Features: Sliding window of historical values
- Hyperparameters:
  - Units: 32-128 per layer
  - Dropout: 0.1-0.3
  - Window Size: Configurable
- Best for: Long-term dependencies and complex patterns
- File: `lstm_model.h5`

### ANN (Artificial Neural Network)
- Architecture: Deep feed-forward neural network
- Layers: Dense layers with ReLU activation
- Features: Automated architecture optimization
- Regularization: L1/L2 with early stopping
- File: `ann_model.h5`

### SVR (Support Vector Regression)
- Kernels: RBF, Linear, Polynomial
- Optimization: Grid search for hyperparameters
- Scaling: Automated feature normalization
- Best for: Non-linear regression with outliers
- File: `svr_model.pkl`

### Prophet
- Components: Trend, Seasonality, Holidays
- Changepoint Detection: Automated
- Uncertainty Intervals: Configurable
- Holiday Effects: Customizable calendar
- File: `prophet_model.pkl`

### ETS (Error, Trend, Seasonality)
- Components: Multiplicative/Additive
- Optimization: Maximum Likelihood
- Seasonality: Multiple seasonal periods
- Confidence Intervals: Built-in
- File: `ets_model.pkl`

## 📁 Project Structure

```
DataMiningForecasting/
├── streamlit_app.py      # Main Streamlit interface
├── app.py                # Core application logic
├── models/               # Model implementation
│   ├── lstm.py          # LSTM model class
│   ├── ann.py           # ANN model class
│   ├── svr.py           # SVR model class
│   ├── prophet.py       # Prophet model class
│   └── ets.py           # ETS model class
├── utils/               # Utility functions
│   ├── preprocessing.py # Data preprocessing
│   ├── evaluation.py    # Model evaluation
│   └── visualization.py # Plotting functions
├── data/                # Data directory
│   ├── train_data.csv  # Training dataset
│   └── test_data.csv   # Testing dataset
└── saved_models/       # Trained model files
    ├── lstm_model.h5
    ├── ann_model.h5
    ├── svr_model.pkl
    ├── prophet_model.pkl
    └── ets_model.pkl
```

## 🔧 Technical Stack

- **Frontend**: 
  - Streamlit 1.x
  - Plotly for interactive visualizations
  - Custom CSS for styling

- **Backend**: 
  - Python 3.8+
  - FastAPI for API endpoints
  - Redis for caching (optional)

- **ML/DL Libraries**: 
  - TensorFlow 2.x / Keras
  - scikit-learn 1.0+
  - Facebook Prophet
  - statsmodels
  - PyTorch (optional for custom models)

- **Data Processing**: 
  - pandas >= 1.3.0
  - numpy >= 1.20.0
  - scipy >= 1.7.0

- **Visualization**: 
  - matplotlib >= 3.4.0
  - plotly >= 5.0.0
  - seaborn >= 0.11.0

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create your feature branch:
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```
4. Make your changes and add tests
5. Run tests:
   ```bash
   pytest tests/
   ```
6. Commit your changes:
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
7. Push to the branch:
   ```bash
   git push origin feature/AmazingFeature
   ```
8. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 👥 Authors

- Talal Muzaffar - *Initial work* - [GitHub](https://github.com/talalmuzaffar)

## 🙏 Acknowledgments

- TensorFlow team for the comprehensive deep learning framework
- Facebook Research for the robust Prophet forecasting tool
- Streamlit team for the excellent web application framework
- The open-source community for various dependencies

## 📞 Support

- For bugs and features, please [open an issue](https://github.com/talalmuzaffar/DataMiningForecasting/issues)
- For security issues, please email directly
- For usage questions, use [GitHub Discussions](https://github.com/talalmuzaffar/DataMiningForecasting/discussions)

## 📚 Documentation

Detailed documentation is available in the `/docs` directory:
- API Reference
- Model Documentation
- Configuration Guide
- Deployment Guide
- Contributing Guidelines
