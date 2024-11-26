from flask import Flask, render_template, request, jsonify, redirect, url_for
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import datetime
import os
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Input
from generate_graph import calculate_combined_performance

app = Flask(__name__)

DATA_DIR = "sp500_data"
MODEL_DIR = "models"

def load_data(filename, sequence_length):
    # Loading data
    data = pd.read_csv(filename)
    print(f"Data loaded from {filename}: {data.head()}")  # Debugging line
    
    if 'Close' not in data.columns:
        raise ValueError(f"The 'Close' column is missing in the data from {filename}")

    data = data[['Close']].values
    print(f"Data shape: {data.shape}")  # Debugging line

    if len(data) == 0:
        raise ValueError(f"No data found in {filename}")

    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    print(f"Scaled data shape: {scaled_data.shape}")  # Debugging line

    # Training
    x, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        x.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])
    
    x, y = np.array(x), np.array(y)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    
    return x, y, scaler, data

def build_rnn(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape)) 
    model.add(LSTM(units=50, return_sequences=True))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(x_train, y_train, model_filepath, epochs=5, batch_size=32):
    if os.path.exists(model_filepath):
        print(f"Loading model from {model_filepath}")
        model = load_model(model_filepath)
    else:
        print(f"Training new model for {model_filepath}")
        model = build_rnn((x_train.shape[1], 1))
        model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        
        # Model saving
        model.save(model_filepath.replace('.h5', '.keras'))  
    return model

def predict_future(model, data, scaler, future_days):
    last_sequence = data[-60:]  # 60 days
    last_sequence_scaled = scaler.transform(last_sequence)

    # Predict future prices
    predicted_prices = []
    for _ in range(future_days):
        x_test = np.array([last_sequence_scaled])
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        predicted_price = model.predict(x_test)
        predicted_prices.append(predicted_price[0, 0])
        
        last_sequence_scaled = np.append(last_sequence_scaled[1:], [[predicted_price[0, 0]]], axis=0)

    predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))
    return predicted_prices.flatten()

def calculate_risk_factor(predicted_prices, actual_prices):
    min_length = min(len(predicted_prices), len(actual_prices))
    errors = predicted_prices[:min_length] - actual_prices[:min_length]
    return np.std(errors)

def process_stock(filepath, sequence_length, future_days):
    ticker = os.path.basename(filepath).split(".")[0]
    model_filepath = os.path.join(MODEL_DIR, f"{ticker}.h5")
    x_data, y_data, scaler, original_data = load_data(filepath, sequence_length)
    model = train_model(x_data, y_data, model_filepath)
    predicted_prices = predict_future(model, original_data, scaler, future_days)
    
    risk_factor = calculate_risk_factor(predicted_prices, y_data[-len(predicted_prices):])
    growth = (predicted_prices[-1] - original_data[-1, 0]) / original_data[-1, 0]
    return ticker, growth, risk_factor

def evaluate_stocks_parallel(future_date):
    sequence_length = 60
    future_days = (future_date - datetime.datetime.now()).days

    if future_days <= 0:
        return []

    stock_files = [os.path.join(DATA_DIR, filename) for filename in os.listdir(DATA_DIR) if filename.endswith(".csv")]

    with ThreadPoolExecutor() as executor:
        results = executor.map(lambda filepath: process_stock(filepath, sequence_length, future_days), stock_files)

    stock_performance = list(results)
    stock_performance.sort(key=lambda x: x[1] / x[2], reverse=True)
    return stock_performance[:10]  # Return top 10 stocks

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        return redirect(url_for('get_best_stocks'), code=302)
    else:
        performance_data = calculate_combined_performance()
        available_tickers = [os.path.splitext(filename)[0] for filename in os.listdir(DATA_DIR) if filename.endswith('.csv')]
        return render_template('index.html', future_date=datetime.datetime.now().strftime('%Y-%m-%d'), available_tickers=available_tickers, performance_data=performance_data)

@app.route('/get_best_stocks', methods=['POST'])
def get_best_stocks():
    data = request.json
    future_date_str = data.get('future_date', datetime.datetime.now().strftime('%Y-%m-%d'))
    future_date = datetime.datetime.strptime(future_date_str, '%Y-%m-%d')
    best_stocks = evaluate_stocks_parallel(future_date)
    return jsonify({'best_stocks': best_stocks})

@app.route('/get_portfolio_performance', methods=['POST'])
def get_portfolio_performance():
    try:
        data = request.get_json()
        future_date_str = data.get('future_date', datetime.datetime.now().strftime('%Y-%m-%d'))
        selected_stocks = data.get('selected_stocks', [])
        future_date = datetime.datetime.strptime(future_date_str, '%Y-%m-%d')

        if not selected_stocks:
            return jsonify({'error': 'No stocks selected'}), 400

        stock_files = [os.path.join(DATA_DIR, f"{stock['ticker']}.csv") for stock in selected_stocks]
        stocks = []

        total_allocation = sum(stock['allocation'] for stock in selected_stocks)
        if not (99.99 <= total_allocation <= 100.01):
            return jsonify({'error': 'Total allocation must equal 100%'}), 400

        for idx, filepath in enumerate(stock_files):
            if os.path.exists(filepath):
                stock = selected_stocks[idx]
                ticker, growth, risk_factor = process_stock(filepath, sequence_length=60, future_days=(future_date - datetime.datetime.now()).days)
                allocation = stock['allocation'] / 100 
                weighted_growth = growth * allocation 
                stocks.append({'ticker': ticker, 'growth': weighted_growth, 'risk_factor': risk_factor, 'allocation': allocation})

        if not stocks:
            raise ValueError('No valid stock data available')

        combined_growth = sum(stock['growth'] for stock in stocks) * 100 

        return jsonify({
            'stocks': stocks,
            'combined_growth': combined_growth
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    
@app.route('/get_stock_tickers', methods=['GET'])
def get_stock_tickers():
    stock_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    tickers = [f.split('.')[0] for f in stock_files]
    return jsonify({'tickers': tickers})

if __name__ == '__main__':
    app.run(debug=True)
