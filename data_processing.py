# data_processing.py
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import matplotlib.pyplot as plt

def download_data():
    """Fetch latest Bitcoin data with volume"""
    df = yf.download('BTC-USD', start='2014-09-17', end=pd.Timestamp.today().strftime('%Y-%m-%d'))
    df.to_csv('bitcoin_data.csv')
    return df

def calculate_rsi(data, window=14):
    """Compute Relative Strength Index (RSI)"""
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def preprocess_data(df, lookback=90):
    """Create time-series sequences with multiple features"""
    # Calculate technical indicators
    df['RSI'] = calculate_rsi(df)
    df = df.dropna()
    
    # Select features
    features = ['Close', 'Volume', 'RSI']
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(df[features])
    
    # Create sequences
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, :])
        y.append(scaled_data[i, 0])  # Predict Close price
    
    X = np.array(X)
    y = np.array(y)
    
    # Save artifacts
    joblib.dump(scaler, 'scaler.pkl')
    return X, y, scaler

def build_model(input_shape):
    """Enhanced LSTM architecture with dropout"""
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(X, y):
    """Train with early stopping"""
    model = build_model((X.shape[1], X.shape[2]))
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    history = model.fit(
        X, y,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )
    
    model.save('btc_predictor.h5')
    return model, history

def evaluate_model(model, X_test, y_test, scaler):
    """Enhanced evaluation with visualization"""
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Inverse transform Close prices
    close_scaler = MinMaxScaler().fit(df[['Close']].values)
    y_pred_actual = close_scaler.inverse_transform(y_pred)
    y_test_actual = close_scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Calculate metrics
    mae = np.mean(np.abs(y_pred_actual - y_test_actual))
    rmse = np.sqrt(np.mean((y_pred_actual - y_test_actual)**2))
    
    print(f"\nValidation MAE: ${mae:.2f}")
    print(f"Validation RMSE: ${rmse:.2f}")
    
    # Plot predictions
    plt.figure(figsize=(12,6))
    plt.plot(y_test_actual[-100:], label='Actual Price')
    plt.plot(y_pred_actual[-100:], label='Predicted Price', linestyle='--')
    plt.title('Actual vs Predicted Prices (Last 100 Days)')
    plt.xlabel('Time Step')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Download and preprocess data
    df = download_data()
    X, y, scaler = preprocess_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        shuffle=False
    )
    
    # Train model
    model, history = train_model(X_train, y_train)
    
    # Evaluate
    evaluate_model(model, X_test, y_test, scaler)
    
    # Plot training history
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()