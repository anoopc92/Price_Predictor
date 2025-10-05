import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Stock & Crypto Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("📈 Stock & Crypto Price Predictor")
st.markdown("**AI-powered price forecasting using LSTM neural networks**")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Asset type selection
    asset_type = st.selectbox(
        "Select Asset Type",
        ["Stock", "Cryptocurrency"]
    )
    
    # Popular symbols based on asset type
    if asset_type == "Stock":
        popular_symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META", "JPM", "V", "WMT"]
        symbol_suffix = ""
    else:
        popular_symbols = ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD", "SOL-USD", "DOGE-USD", "DOT-USD", "MATIC-USD", "AVAX-USD"]
        symbol_suffix = "-USD"
    
    # Symbol selection
    col1, col2 = st.columns([3, 1])
    with col1:
        symbol_input = st.selectbox(
            "Select or Enter Symbol",
            popular_symbols,
            index=0
        )
    with col2:
        custom_symbol = st.text_input("Custom", "", key="custom")
    
    symbol = custom_symbol if custom_symbol else symbol_input
    
    # Date range selection
    st.subheader("📅 Date Range")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # 2 years default
    
    date_range = st.date_input(
        "Select Training Period",
        value=(start_date, end_date),
        max_value=end_date
    )
    
    # Model parameters
    st.subheader("🧠 Model Parameters")
    prediction_days = st.slider("Prediction Horizon (days)", 1, 30, 7)
    lookback_days = st.slider("Lookback Window (days)", 30, 120, 60)
    epochs = st.slider("Training Epochs", 10, 100, 50)
    
    # Train button
    train_button = st.button("🚀 Train & Predict", type="primary", use_container_width=True)

# Helper functions
@st.cache_data(ttl=3600)
def fetch_data(symbol, start, end):
    """Fetch historical data from Yahoo Finance"""
    try:
        data = yf.download(symbol, start=start, end=end, progress=False)
        if data.empty:
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def prepare_data(data, lookback=60):
    """Prepare data for LSTM model"""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler

def build_lstm_model(lookback):
    """Build LSTM model architecture"""
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def calculate_metrics(actual, predicted):
    """Calculate performance metrics"""
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return rmse, mae, mape

# Main application logic
if train_button:
    if len(date_range) != 2:
        st.error("Please select both start and end dates.")
    else:
        start, end = date_range
        
        with st.spinner(f"Fetching data for {symbol}..."):
            data = fetch_data(symbol, start, end)
        
        if data is None or len(data) < lookback_days + 30:
            st.error("Insufficient data. Please select a longer date range or different symbol.")
        else:
            # Display data info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Days", len(data))
            with col2:
                st.metric("Latest Close", f"${data['Close'].iloc[-1]:.2f}")
            with col3:
                change = ((data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100
                st.metric("Period Change", f"{change:.2f}%")
            with col4:
                st.metric("Avg Volume", f"{data['Volume'].mean()/1e6:.2f}M")
            
            # Split data
            split_idx = int(len(data) * 0.8)
            train_data = data[:split_idx]
            test_data = data[split_idx:]
            
            # Prepare data
            with st.spinner("Preparing data..."):
                X, y, scaler = prepare_data(data, lookback_days)
                train_size = int(len(X) * 0.8)
                X_train, X_test = X[:train_size], X[train_size:]
                y_train, y_test = y[:train_size], y[train_size:]
            
            # Build and train model
            with st.spinner("Training LSTM model... This may take a few minutes."):
                model = build_lstm_model(lookback_days)
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                class StreamlitCallback(keras.callbacks.Callback):
                    def on_epoch_end(self, epoch, logs=None):
                        progress = (epoch + 1) / epochs
                        progress_bar.progress(progress)
                        status_text.text(f"Epoch {epoch + 1}/{epochs} - Loss: {logs['loss']:.6f}")
                
                history = model.fit(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=32,
                    validation_split=0.1,
                    verbose=0,
                    callbacks=[StreamlitCallback()]
                )
                
                progress_bar.empty()
                status_text.empty()
            
            st.success("✅ Model training completed!")
            
            # Make predictions
            predictions = model.predict(X_test, verbose=0)
            predictions = scaler.inverse_transform(predictions)
            actual = scaler.inverse_transform(y_test.reshape(-1, 1))
            
            # Calculate metrics
            rmse, mae, mape = calculate_metrics(actual, predictions)
            
            # Display metrics
            st.subheader("📊 Model Performance")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("RMSE", f"${rmse:.2f}")
            with col2:
                st.metric("MAE", f"${mae:.2f}")
            with col3:
                st.metric("MAPE", f"{mape:.2f}%")
            
            # Plot results
            st.subheader("📈 Historical Data & Predictions")
            
            fig = make_subplots(
                rows=2, cols=1,
                row_heights=[0.7, 0.3],
                subplot_titles=('Price Prediction', 'Training Loss'),
                vertical_spacing=0.1
            )
            
            # Price plot
            test_dates = data.index[split_idx + lookback_days:]
            
            fig.add_trace(
                go.Scatter(
                    x=data.index[:split_idx],
                    y=data['Close'][:split_idx],
                    name='Training Data',
                    line=dict(color='gray', width=1)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=test_dates,
                    y=actual.flatten(),
                    name='Actual Price',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=test_dates,
                    y=predictions.flatten(),
                    name='Predicted Price',
                    line=dict(color='red', width=2, dash='dash')
                ),
                row=1, col=1
            )
            
            # Loss plot
            fig.add_trace(
                go.Scatter(
                    y=history.history['loss'],
                    name='Training Loss',
                    line=dict(color='orange')
                ),
                row=2, col=1
            )
            
            if 'val_loss' in history.history:
                fig.add_trace(
                    go.Scatter(
                        y=history.history['val_loss'],
                        name='Validation Loss',
                        line=dict(color='green')
                    ),
                    row=2, col=1
                )
            
            fig.update_xaxes(title_text="Date", row=1, col=1)
            fig.update_xaxes(title_text="Epoch", row=2, col=1)
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="Loss", row=2, col=1)
            
            fig.update_layout(
                height=800,
                showlegend=True,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Future predictions
            st.subheader("🔮 Future Price Forecast")
            
            last_sequence = scaled_data = scaler.transform(data['Close'].values[-lookback_days:].reshape(-1, 1))
            future_predictions = []
            
            for _ in range(prediction_days):
                next_pred = model.predict(last_sequence.reshape(1, lookback_days, 1), verbose=0)
                future_predictions.append(next_pred[0, 0])
                last_sequence = np.append(last_sequence[1:], next_pred).reshape(-1, 1)
            
            future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
            
            # Create future dates
            last_date = data.index[-1]
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=prediction_days)
            
            # Plot future predictions
            fig_future = go.Figure()
            
            fig_future.add_trace(
                go.Scatter(
                    x=data.index[-60:],
                    y=data['Close'][-60:],
                    name='Historical',
                    line=dict(color='blue', width=2)
                )
            )
            
            fig_future.add_trace(
                go.Scatter(
                    x=future_dates,
                    y=future_predictions.flatten(),
                    name='Forecast',
                    line=dict(color='red', width=2, dash='dash'),
                    mode='lines+markers'
                )
            )
            
            fig_future.update_layout(
                title=f"{symbol} - {prediction_days} Day Price Forecast",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig_future, use_container_width=True)
            
            # Forecast table
            forecast_df = pd.DataFrame({
                'Date': future_dates.strftime('%Y-%m-%d'),
                'Predicted Price': [f"${p[0]:.2f}" for p in future_predictions],
                'Change from Today': [f"{((p[0] - data['Close'].iloc[-1]) / data['Close'].iloc[-1] * 100):.2f}%" for p in future_predictions]
            })
            
            st.dataframe(forecast_df, use_container_width=True, hide_index=True)
            
            # Download predictions
            csv = forecast_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Forecast CSV",
                data=csv,
                file_name=f"{symbol}_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

else:
    # Welcome screen
    st.info("👈 Configure your parameters in the sidebar and click '🚀 Train & Predict' to start!")
    
    st.markdown("""
    ### 🎯 Features
    - **Multiple Assets**: Support for stocks and cryptocurrencies
    - **LSTM Neural Network**: Deep learning for time-series prediction
    - **Interactive Charts**: Beautiful visualizations with Plotly
    - **Customizable**: Adjust lookback window, prediction horizon, and training epochs
    - **Performance Metrics**: RMSE, MAE, and MAPE evaluation
    - **Future Forecasts**: Predict prices for up to 30 days ahead
    
    ### 📖 How to Use
    1. Select asset type (Stock or Crypto)
    2. Choose or enter a symbol
    3. Set your preferred date range for training
    4. Adjust model parameters
    5. Click "Train & Predict" and wait for results
    
    ### ⚠️ Disclaimer
    This tool is for educational purposes only. Predictions are based on historical data and should not be used as financial advice.
    Always conduct thorough research before making investment decisions.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 1rem;'>
    <p>Built with Streamlit, TensorFlow & ❤️ | Data from Yahoo Finance</p>
</div>
""", unsafe_allow_html=True)
