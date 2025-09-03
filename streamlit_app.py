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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.impute import SimpleImputer
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


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


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from sklearn.impute import SimpleImputer
import tensorflow as tf

# --------------------------
# Load model
# --------------------------
model = tf.keras.models.load_model("multivariate_lstm_model.keras")

# --------------------------
# CSV Upload
# --------------------------
uploaded_file = st.file_uploader("Upload CSV for prediction", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Input Data")
    st.dataframe(df)

    # --------------------------
    # Preprocessing
    # --------------------------
    volatility_features = ['Rolling Volatility', 'Bollinger_Width']
    pt_volatility = PowerTransformer(method='box-cox', standardize=False)
    df[volatility_features] = pt_volatility.fit_transform(df[volatility_features])

    imputer = SimpleImputer(missing_values=np.nan)
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    target_scaler = MinMaxScaler(feature_range=(0,1))
    df_scaled[['Open','Close']] = target_scaler.fit_transform(df[['Open','Close']])

    # --------------------------
    # LSTM prediction
    # --------------------------
    timesteps = 20
    features = df_scaled.columns.tolist()
    predictions = []

    if len(df_scaled) < timesteps:
        st.error(f"CSV must have at least {timesteps} rows for prediction.")
    else:
        # If exact timesteps, just simulate one prediction
        if len(df_scaled) == timesteps:
            X_input = df_scaled[features].iloc[:timesteps].values
            X_input = X_input.reshape(1, timesteps, len(features))
            pred_scaled = model.predict(X_input, verbose=0)
            pred = target_scaler.inverse_transform(pred_scaled)
            predictions.append(pred[0])
        else:
            # Normal multi-row prediction
            for i in range(timesteps, len(df_scaled)):
                X_input = df_scaled[features].iloc[i-timesteps:i].values
                X_input = X_input.reshape(1, timesteps, len(features))
                pred_scaled = model.predict(X_input, verbose=0)
                pred = target_scaler.inverse_transform(pred_scaled)
                predictions.append(pred[0])

        # --------------------------
        # Prepare results table
        # --------------------------
        pred_df = pd.DataFrame(predictions, columns=['Predicted Open','Predicted Close'])

        # Previous day actual
        prev_actual = df[['Open','Close']].iloc[timesteps-1:-1].reset_index(drop=True)
        # If only one prediction, prev_actual is single row
        if len(pred_df) == 1:
            prev_actual = df[['Open','Close']].iloc[timesteps-1:timesteps].reset_index(drop=True)

        pred_df['Prev Day Open'] = prev_actual['Open']
        pred_df['Prev Day Close'] = prev_actual['Close']

        st.subheader("Predictions with Previous Day Actuals")
        st.dataframe(pred_df)
