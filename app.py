import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, r2_score

def fetch_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data[['Open', 'High', 'Low', 'Close', 'Volume']]

def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0,1))
    data_scaled = scaler.fit_transform(data)
    return data_scaled, scaler

def create_sequences(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i+time_step), :])
        y.append(data[i + time_step, :])
    return np.array(X), np.array(y)

def build_lstm_model():
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(60, 5)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(5)  # Predicting 5 features
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def predict_next_days(model, last_60_days, scaler, days=7):
    future_predictions = []
    input_seq = last_60_days.reshape(1, 60, 5)
    
    for _ in range(days):
        pred = model.predict(input_seq)[0]
        future_predictions.append(pred.tolist())
        input_seq = np.append(input_seq[:, 1:, :], [[pred]], axis=1)  # Shift window
    
    return scaler.inverse_transform(np.array(future_predictions))

def main():
    st.title("📈 Stock Price Prediction with LSTM")
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL):", "AAPL")
    end_date = st.date_input("Prediction Date:", pd.to_datetime("today").date())
    start_date = end_date - pd.DateOffset(years=4)
    
    if st.button("Fetch Data and Train Model"):
        st.write(f"Fetching data for {ticker} from {start_date} to {end_date}...")
        data = fetch_stock_data(ticker, start_date, end_date)
        
        if data.empty:
            st.error("No data found! Try another ticker.")
            return
        
        st.write("### Last 10 Days of Fetched Data")
        st.write(data.tail(10))
        
        data_scaled, scaler = preprocess_data(data)
        
        X, y = create_sequences(data_scaled)
        X = X.reshape(X.shape[0], X.shape[1], 5)
        
        model = build_lstm_model()
        model.fit(X, y, epochs=20, batch_size=16, verbose=1)
        
        last_60_days = data_scaled[-60:].reshape(1, 60, 5)
        future_prices = predict_next_days(model, last_60_days, scaler, days=7)
        
        future_dates = pd.date_range(start=end_date, periods=7, freq='B')
        predictions_df = pd.DataFrame({
            "Date": future_dates,
            "Predicted Open": future_prices[:, 0],
            "Predicted High": future_prices[:, 1],
            "Predicted Low": future_prices[:, 2],
            "Predicted Close": future_prices[:, 3],
            "Predicted Volume": future_prices[:, 4]
        })
        
        st.write("### Predicted Prices for the Next 7 Days")
        st.write(predictions_df)
        
        # Evaluate model performance
        y_pred = model.predict(X)
        y_pred = scaler.inverse_transform(y_pred)
        y_actual = scaler.inverse_transform(y)
        
        mse = mean_squared_error(y_actual, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_actual, y_pred)
        
        st.write("### Model Performance Metrics")
        st.write(f"Mean Squared Error (MSE): {mse:.4f}")
        st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        st.write(f"R² Score: {r2:.4f}")
        
        # Plot actual vs predicted prices
        st.write("### Actual vs Predicted Prices")
        fig, ax = plt.subplots()
        ax.plot(y_actual[:, 3], label="Actual Close Prices", color="blue")
        ax.plot(y_pred[:, 3], label="Predicted Close Prices", color="red")
        ax.set_xlabel("Days")
        ax.set_ylabel("Price")
        ax.set_title("Stock Price Prediction")
        ax.legend()
        st.pyplot(fig)

if __name__ == "__main__":
    main()
