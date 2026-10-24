"""Paths to the datasets and saved models, resolved relative to the repository root."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'

TRAIN_DATA_PATH = DATA_DIR / 'train_data.csv'
TEST_DATA_PATH = DATA_DIR / 'test_data.csv'

# Model paths are strings because load_model checks the file extension
ETS_MODEL_PATH = str(MODELS_DIR / 'ets_model.pkl')
PROPHET_MODEL_PATH = str(MODELS_DIR / 'prophet_model.pkl')
SVR_MODEL_PATH = str(MODELS_DIR / 'svr_model.pkl')
ANN_MODEL_PATH = str(MODELS_DIR / 'ann_model.h5')
LSTM_MODEL_PATH = str(MODELS_DIR / 'lstm_model.h5')

# Also loaded by the Flask API (app.py); these files are not included in the repository
SARIMAX_MODEL_PATH = str(MODELS_DIR / 'sarimax_model.pkl')
ARIMA_MODEL_PATH = str(MODELS_DIR / 'arima_model.pkl')
HYBRID_MODEL_PATH = str(MODELS_DIR / 'hybrid_model.pkl')
