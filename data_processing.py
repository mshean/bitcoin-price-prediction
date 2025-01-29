# data_processing.py
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import matplotlib.pyplot as plt

def download_data():
    """Fetch Bitcoin data with proper columns"""
    df = yf.download('BTC-USD', start='2014-09-17', end=pd.Timestamp.today().strftime('%Y-%m-%d'))
    df.reset_index(inplace=True)
    return df[['Date', 'Close', 'Volume']]  # Explicitly select columns

def calculate_rsi(data, window=14):
    """Add RSI to DataFrame"""
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    return data

def preprocess_data(df, lookback=90):
    """Process data and return aligned features"""
    df = calculate_rsi(df).dropna()
    
    # Define features and target
    features = ['Close', 'Volume', 'RSI']
    target = 'Close'
    
    # Initialize scalers
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Scale features and target
    scaled_features = feature_scaler.fit_transform(df[features])
    scaled_target = target_scaler.fit_transform(df[[target]])
    
    # Save scalers and processed data
    joblib.dump(feature_scaler, 'feature_scaler.pkl')
    joblib.dump(target_scaler, 'target_scaler.pkl')
    df[features].to_csv('bitcoin_data_processed.csv', index=False)
    
    # Create sequences
    X, y = [], []
    for i in range(lookback, len(scaled_features)):
        X.append(scaled_features[i-lookback:i, :])
        y.append(scaled_target[i, 0])
    
    return np.array(X), np.array(y), feature_scaler, target_scaler

def build_model(input_shape):
    """LSTM model with dropout"""
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

if __name__ == "__main__":
    # Download and process data
    df = download_data()
    X, y, feature_scaler, target_scaler = preprocess_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        shuffle=False
    )
    
    # Train model
    model = build_model((X_train.shape[1], X_train.shape[2]))
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )
    model.save('btc_predictor.h5')