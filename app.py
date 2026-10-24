"""Flask REST API for the forecasting models; run with `python app.py`."""
from data_mining_forecasting.api import app

if __name__ == '__main__':
    app.run(debug=True)
