import os
import logging
import warnings

# -------------------------
# Suppress TensorFlow logs
# -------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # hide INFO, WARNING, ERROR
logging.getLogger('tensorflow').setLevel(logging.FATAL)

# -------------------------
# Suppress all warnings
# -------------------------
warnings.filterwarnings("ignore", category=UserWarning)  # Streamlit "missing ScriptRunContext" is a UserWarning
warnings.filterwarnings("ignore")  # optional: ignore all other warnings

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go


st.set_page_config(
    page_title="Stock Dashboard",
    layout="wide"  # makes the page full-width
)
st.title("🎈 My Thivya app")
st.title("Interactive EDA Dashboard")

# Load dataset
@st.cache_data  # caches the dataset for faster reload
def load_data():
    data = pd.read_csv("stock_df.csv")  # replace with your CSV path
    return data

data = load_data()

# -------------------------
# Select ticker
# -------------------------
tickers = data['Ticker'].unique()
selected_ticker = st.selectbox("Select Ticker", tickers)

ticker_data = data[data['Ticker'] == selected_ticker]

# -------------------------
# Create side-by-side columns
# -------------------------
col1, col2 = st.columns([4, 2])  # left column bigger

# -------------------------
# Left Column: Open & Close Prices
# -------------------------
with col1:
    st.subheader(f"{selected_ticker} Open & Close Prices")
    
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=ticker_data['Date'],
        y=ticker_data['Open'],
        mode='lines',
        name='Open'
    ))
    fig1.add_trace(go.Scatter(
        x=ticker_data['Date'],
        y=ticker_data['Close'],
        mode='lines',
        name='Close'
    ))

    max_date = ticker_data['Date'].max()

    fig1.update_layout(
        title="Open vs Close Prices",
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(step="all")  # shows entire range
            ])
        ),
        rangeslider=dict(visible=False),  # optional
        type="date",
        range=[ticker_data['Date'].min(), max_date]  # show only dataset range
    ),

    template='plotly_white',
    height=500 
)
    st.plotly_chart(fig1, use_container_width=True)

# -------------------------
# Right Column: Volume
# -------------------------
with col2:
    st.subheader(f"{selected_ticker} Volume")
    
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=ticker_data['Date'],
        y=ticker_data['Volume'],
        name='Volume',
        marker_color='orange'
    ))

    fig2.update_layout(
        title="Trading Volume",
        xaxis_title='Date',
        yaxis_title='Volume',
        xaxis=dict(rangeslider=dict(visible=True)),
        template='plotly_white',
        height=500  # taller figure
    )
    
    st.plotly_chart(fig2, use_container_width=True)

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
# --------------------------
# Load model
# --------------------------
model = tf.keras.models.load_model("multivariate_lstm_model.keras")

# --------------------------
# Upload CSV
# --------------------------
uploaded_file = st.file_uploader("Upload CSV for prediction", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    st.subheader("Input Data")
    st.dataframe(df)

    # --------------------------
    # Parameters
    # --------------------------
    features = ['Open', 'Close', 'Volume']  # input features
    target_cols = ['Open', 'Close']
    timesteps = 20

    # --------------------------
    # Create sequences for LSTM
    # --------------------------
    X_input = []
    for i in range(timesteps, len(df)):
        X_input.append(df[features].iloc[i-timesteps:i].values)
    X_input = np.array(X_input)

    # --------------------------
    # Scale features
    # --------------------------
    scaler_X = MinMaxScaler()
    X_input_reshaped = X_input.reshape(-1, len(features))
    X_input_scaled = scaler_X.fit_transform(X_input_reshaped).reshape(X_input.shape)

    # --------------------------
    # Predict
    # --------------------------
    predicted = model.predict(X_input_scaled)  # shape: (num_samples, 2)

    # --------------------------
    # Prepare table
    # --------------------------
    result_df = df.iloc[timesteps:].copy()  # dates corresponding to predictions
    # Add predictions
    result_df[['Predicted Open', 'Predicted Close']] = predicted

    # Add previous day actual values
    prev_actual = df[target_cols].iloc[timesteps-1:-1].reset_index(drop=True)
    result_df['Prev Day Open'] = prev_actual['Open']
    result_df['Prev Day Close'] = prev_actual['Close']

    # Select columns for display
    display_cols = ['Date', 'Predicted Open', 'Predicted Close', 'Prev Day Open', 'Prev Day Close']
    st.subheader("Predictions with Previous Day Actuals")
    st.dataframe(result_df[display_cols])
