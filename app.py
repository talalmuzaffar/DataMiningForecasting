from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
# from keras.models import load_model
import tensorflow as tf
app = Flask(__name__)
CORS(app)

# Load the models
sarimax_model = joblib.load('sarimax_model.pkl')
arima_model = joblib.load('arima_model.pkl')
ets_model = joblib.load('ets_model.pkl')
prophet_model = joblib.load('prophet_model.pkl')
svr_model = joblib.load('svr_model.pkl')
hybrid_model = joblib.load('hybrid_model.pkl')
ann_model = tf.keras.models.load_model('ann_model.h5')
lstm_model = tf.keras.models.load_model('lstm_model.h5')


@app.route('/forecast_sarimax', methods=['POST'])
def forecast_sarimax():
    data = request.json
    exogenous_columns = ['total load forecast']
    
    # Preprocess input data
    df = pd.DataFrame(data)
    
    # Define the number of steps to forecast
    forecast_steps = 10
    
    # Forecast
    forecast = sarimax_model.predict(start=len(df), end=len(df) + forecast_steps - 1, exog=df[exogenous_columns])
    
    return jsonify({'forecast': forecast.tolist()})

@app.route('/forecast_arima', methods=['POST'])
def forecast_arima():
    data = request.json
    
    # Preprocess input data
    df = pd.DataFrame(data)
    
    # Define the number of steps to forecast
    forecast_steps = 10
    
    # Forecast
    forecast = arima_model.predict(start=len(df), end=len(df) + forecast_steps - 1)
    
    return jsonify({'forecast': forecast.tolist()})

@app.route('/forecast_ets', methods=['POST'])
def forecast_ets():
    data = request.json
    
    # Preprocess input data
    df = pd.DataFrame(data)
    
    # Define the number of steps to forecast
    forecast_steps = 10
    
    # Forecast
    forecast = ets_model.predict(start=len(df), end=len(df) + forecast_steps - 1)
    
    return jsonify({'forecast': forecast.tolist()})


@app.route('/forecast_prophet', methods=['POST'])

def forecast_prophet():
    data = request.json
    
    # Preprocess input data
    df = pd.DataFrame(data)

    forecast_steps = 10
    
    # Forecast
    future = prophet_model.make_future_dataframe(periods=forecast_steps)
    forecast = prophet_model.predict(future)
    forecast_values = forecast[['ds', 'yhat']].tail(forecast_steps)['yhat'].values
    
    return jsonify({'forecast': forecast_values.tolist()})




@app.route('/forecast_svr', methods=['POST'])
def forecast_svr():
    data = request.json
    
    # Preprocess input data
    df = pd.DataFrame(data)
    
    # Forecast using SVR model
    forecast = svr_model.predict(df)
    
    return jsonify({'forecast': forecast.tolist()})



@app.route('/forecast_ann', methods=['POST'])
def forecast_ann():
    data = request.json
    
    # Preprocess input data
    df = pd.DataFrame(data)
    
    # Forecast using ANN model
    forecast = ann_model.predict(df)
    
    return jsonify({'forecast': forecast.tolist()})



@app.route('/forecast_lstm', methods=['POST'])
def forecast_lstm():
    data = request.json
    
    # Preprocess input data
    df = pd.DataFrame(data)
    
    # Forecast using ANN model
    forecast = ann_model.predict(df)
    
    return jsonify({'forecast': forecast.tolist()})


@app.route('/forecast_hybrid', methods=['POST'])
def forecast_hybrid():
    data = request.json
    
    # Preprocess input data
    df = pd.DataFrame(data)
    
    # Define the number of steps to forecast
    forecast_steps = 10
    
    # Forecast
    forecast = hybrid_model.predict(start=len(df), end=len(df) + forecast_steps - 1)
    
    return jsonify({'forecast': forecast.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
