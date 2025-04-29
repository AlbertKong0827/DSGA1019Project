from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def lstm_forecast(df, window_size=12, plot=True):
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    target_col = 'avg_adjusted_delinquency_index'
    values = df[[target_col]].values

    # Normalize
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(values)

    def create_sequences(data, window_size=12):
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:i+window_size])
            y.append(data[i+window_size])
        return np.array(X), np.array(y)

    X, y = create_sequences(scaled_values, window_size)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential([
        LSTM(64, activation='tanh', input_shape=(window_size, 1)),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), verbose=0)

    y_pred = model.predict(X_test)

    # Reshape before inverse transform
    y_pred_rescaled = scaler.inverse_transform(y_pred.reshape(-1, 1))
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Evaluation metrics
    rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
    mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
    mape = np.mean(np.abs((y_test_rescaled - y_pred_rescaled) / np.maximum(np.abs(y_test_rescaled), 1e-6))) * 100

    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.2f}%")

    # Prediction dates aligned
    prediction_dates = df['date'].iloc[split+window_size:].reset_index(drop=True)

    if plot:
        plt.figure(figsize=(10,6))
        plt.plot(prediction_dates, y_test_rescaled.flatten(), label='Actual')
        plt.plot(prediction_dates, y_pred_rescaled.flatten(), label='Predicted', linestyle='--')
        plt.legend()
        plt.title('LSTM Forecast: Actual vs Predicted')
        plt.xlabel('Date')
        plt.ylabel('Adjusted Delinquency Index')
        plt.grid(True)
        plt.show()

    return y_pred_rescaled.flatten(), prediction_dates



