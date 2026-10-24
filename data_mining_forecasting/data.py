"""Loading the train and test datasets."""
import pandas as pd

from data_mining_forecasting.config import TEST_DATA_PATH, TRAIN_DATA_PATH


# Load test and train data
def load_data():
    test_data = pd.read_csv(TEST_DATA_PATH, index_col=0, parse_dates=True)
    test_actual = test_data['total load actual']
    train_data = pd.read_csv(TRAIN_DATA_PATH, index_col=0, parse_dates=True)
    train_actual = train_data['total load actual']
    return test_data, test_actual, train_data, train_actual
