# predict.py
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import matplotlib.pyplot as plt

def load_artifacts():
    """Load model and scalers"""
    model = load_model('btc_predictor.h5')
    feature_scaler = joblib.load('feature_scaler.pkl')
    target_scaler = joblib.load('target_scaler.pkl')
    df = pd.read_csv('bitcoin_data_processed.csv')
    return model, feature_scaler, target_scaler, df

def predict_future(model, feature_scaler, target_scaler, df, num_days=7):
    """Generate predictions using the trained model"""
    lookback = model.input_shape[1]
    features = ['Close', 'Volume', 'RSI']
    
    # Get last lookback days of data
    input_data = df[features].iloc[-lookback:].values
    
    # Scale features
    scaled_input = feature_scaler.transform(input_data).reshape(1, lookback, 3)
    
    predictions = []
    for _ in range(num_days):
        # Predict next value
        pred = model.predict(scaled_input, verbose=0)[0][0]
        predictions.append(pred)
        
        # Update input sequence with dummy values
        new_row = np.array([pred, 0, 0]).reshape(1, 1, 3)  # 3D array (1,1,3)
        scaled_input = np.concatenate([scaled_input[:, 1:, :], new_row], axis=1)
    
    # Inverse scale predictions
    return target_scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

if __name__ == "__main__":
    model, feature_scaler, target_scaler, df = load_artifacts()
    predictions = predict_future(model, feature_scaler, target_scaler, df)
    
    print("\nBitcoin Price Predictions:")
    for i, price in enumerate(predictions):
        print(f"Day {i+1}: ${price[0]:.2f}")