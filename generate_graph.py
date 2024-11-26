import os
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

def calculate_combined_performance():
    data_folder = 'sp500_data'
    model_folder = 'models'
    static_folder = 'static'

    train_start_date = '2022-10-01'
    train_end_date = '2023-07-31'

    display_start_date = '2023-01-01'
    display_end_date = '2023-07-31'

    csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]

    random_stocks = random.sample(csv_files, 10)

    combined_data = pd.DataFrame()

    random_stock_tickers = []

    for stock in random_stocks:
        stock_ticker = stock.replace('.csv', '')
        random_stock_tickers.append(stock_ticker)
        stock_data = pd.read_csv(os.path.join(data_folder, stock), index_col='Date', parse_dates=True)
        stock_data = stock_data[(stock_data.index >= train_start_date) & (stock_data.index <= train_end_date)]
        combined_data[stock_ticker] = stock_data['Close']

    combined_data['Average_Actual_Price'] = combined_data.mean(axis=1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    combined_data_scaled = pd.DataFrame(scaler.fit_transform(combined_data[random_stock_tickers]), index=combined_data.index, columns=random_stock_tickers)

    def prepare_data_for_prediction(data, window_size):
        X = []
        for i in range(len(data) - window_size):
            X.append(data[i:i+window_size])
        return np.array(X)

    window_size = 60

    predicted_data = pd.DataFrame(index=combined_data.index[window_size:], columns=random_stock_tickers)

    for stock_ticker in random_stock_tickers:
        model = load_model(os.path.join(model_folder, f'{stock_ticker}.h5'))
        X_test = prepare_data_for_prediction(combined_data_scaled[stock_ticker].values, window_size)
        predicted_prices_scaled = model.predict(X_test).flatten()
        
        scaler_stock = MinMaxScaler(feature_range=(0, 1))
        scaler_stock.fit(combined_data[[stock_ticker]])
        
        predicted_prices = scaler_stock.inverse_transform(predicted_prices_scaled.reshape(-1, 1)).flatten()
        predicted_data[stock_ticker] = predicted_prices

    predicted_data['Average_Predicted_Price'] = predicted_data.mean(axis=1)

    # Filter the data for the display period
    combined_data_display = combined_data[(combined_data.index >= display_start_date) & (combined_data.index <= display_end_date)]
    predicted_data_display = predicted_data[(predicted_data.index >= display_start_date) & (predicted_data.index <= display_end_date)]

    # Calculate the initial prices for the display period
    initial_actual_price = combined_data_display['Average_Actual_Price'].iloc[0]
    initial_predicted_price = predicted_data_display['Average_Predicted_Price'].iloc[0]

    # Calculate gain/loss over time based on $2000 investment
    investment_amount = 2000

    combined_data_display['Actual_Gain_Loss'] = (combined_data_display['Average_Actual_Price'] / initial_actual_price - 1) * investment_amount
    predicted_data_display['Predicted_Gain_Loss'] = (predicted_data_display['Average_Predicted_Price'] / initial_predicted_price - 1) * investment_amount

    # Plot the gain/loss instead of the average price
    plt.figure(figsize=(14, 7))
    plt.plot(combined_data_display.index, combined_data_display['Actual_Gain_Loss'], label='Actual Combined Gain/Loss', color='blue')
    plt.plot(predicted_data_display.index, predicted_data_display['Predicted_Gain_Loss'], label='Predicted Combined Gain/Loss', color='orange')
    plt.title('Actual vs Predicted Combined Gain/Loss (10 Random Stocks)')
    plt.xlabel('Date')
    plt.ylabel('Gain/Loss ($)')
    plt.legend()
    plt.grid(True)

    image_path = os.path.join(static_folder, 'stock_gain_loss_comparison.png')
    plt.savefig(image_path)

    # Return the final gain/loss values and performance difference
    actual_gain_loss = combined_data_display['Actual_Gain_Loss'].iloc[-1]
    predicted_gain_loss = predicted_data_display['Predicted_Gain_Loss'].iloc[-1]
    performance_difference = predicted_gain_loss - actual_gain_loss
    percentage_difference = (performance_difference / actual_gain_loss) * 100 if actual_gain_loss != 0 else float('inf')

    return {
        'actual_gain_loss': actual_gain_loss,
        'predicted_gain_loss': predicted_gain_loss,
        'difference': performance_difference,
        'percentage_difference': percentage_difference
    }

