import streamlit as st
import os
from google.genai import Client

import streamlit as st
import os
import requests
import json

st.title("Gemini 2.0 Flash Content Generator")

# --- Set your API key in Streamlit Secrets ---
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    st.error("GEMINI_API_KEY is missing! Add it in Streamlit Secrets.")
    st.stop()

# --- Dropdown for user topic selection ---
topic = st.selectbox(
    "Choose a topic:",
    ["Artificial Intelligence", "Stock Market", "Cryptocurrency", "Finance"]
)

# Optional user prompt
user_prompt = st.text_area("Or enter your own prompt:")

if st.button("Generate Content"):
    # Construct the prompt
    prompt_text = user_prompt if user_prompt else f"Explain {topic} in simple words"

    # Gemini 2.0 Flash API endpoint
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": API_KEY
    }

    data = {
        "contents": [
            {
                "parts": [
                    {"text": prompt_text}
                ]
            }
        ]
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        result = response.json()

        # Extract the generated text
        generated_text = result["candidates"][0]["content"][0]["text"]

        st.subheader("Generated Content")
        st.write(generated_text)

    except Exception as e:
        st.error(f"Error generating content: {e}")



# import os
# import logging
# import warnings

# # -------------------------
# # Suppress TensorFlow logs
# # -------------------------
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # hide INFO, WARNING, ERROR
# logging.getLogger('tensorflow').setLevel(logging.FATAL)

# # -------------------------
# # Suppress all warnings
# # -------------------------
# warnings.filterwarnings("ignore", category=UserWarning)  # Streamlit "missing ScriptRunContext" is a UserWarning
# warnings.filterwarnings("ignore")  # optional: ignore all other warnings

# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.express as px
# import plotly.graph_objects as go
# from sklearn.impute import SimpleImputer
# import tensorflow as tf
# from sklearn.preprocessing import MinMaxScaler


# st.set_page_config(
#     page_title="Stock Dashboard",
#     layout="wide"  # makes the page full-width
# )
# st.title("🎈 My Thivya app")
# st.title("Interactive EDA Dashboard")

# # Load dataset
# @st.cache_data  # caches the dataset for faster reload
# def load_data():
#     data = pd.read_csv("stock_df.csv")  # replace with your CSV path
#     return data

# data = load_data()

# # -------------------------
# # Select ticker
# # -------------------------
# tickers = data['Ticker'].unique()
# selected_ticker = st.selectbox("Select Ticker", tickers)

# ticker_data = data[data['Ticker'] == selected_ticker]

# # -------------------------
# # Create side-by-side columns
# # -------------------------
# col1, col2 = st.columns([4, 2])  # left column bigger

# # -------------------------
# # Left Column: Open & Close Prices
# # -------------------------
# with col1:
#     st.subheader(f"{selected_ticker} Open & Close Prices")
    
#     fig1 = go.Figure()
#     fig1.add_trace(go.Scatter(
#         x=ticker_data['Date'],
#         y=ticker_data['Open'],
#         mode='lines',
#         name='Open'
#     ))
#     fig1.add_trace(go.Scatter(
#         x=ticker_data['Date'],
#         y=ticker_data['Close'],
#         mode='lines',
#         name='Close'
#     ))

#     max_date = ticker_data['Date'].max()

#     fig1.update_layout(
#         title="Open vs Close Prices",
#         xaxis_title='Date',
#         yaxis_title='Price',
#         xaxis=dict(
#         rangeselector=dict(
#             buttons=list([
#                 dict(count=1, label="1M", step="month", stepmode="backward"),
#                 dict(count=6, label="6M", step="month", stepmode="backward"),
#                 dict(count=1, label="1Y", step="year", stepmode="backward"),
#                 dict(step="all")  # shows entire range
#             ])
#         ),
#         rangeslider=dict(visible=False),  # optional
#         type="date",
#         range=[ticker_data['Date'].min(), max_date]  # show only dataset range
#     ),

#     template='plotly_white',
#     height=500 
# )
#     st.plotly_chart(fig1, use_container_width=True)

# # -------------------------
# # Right Column: Volume
# # -------------------------
# with col2:
#     st.subheader(f"{selected_ticker} Volume")
    
#     fig2 = go.Figure()
#     fig2.add_trace(go.Bar(
#         x=ticker_data['Date'],
#         y=ticker_data['Volume'],
#         name='Volume',
#         marker_color='orange'
#     ))

#     fig2.update_layout(
#         title="Trading Volume",
#         xaxis_title='Date',
#         yaxis_title='Volume',
#         xaxis=dict(rangeslider=dict(visible=True)),
#         template='plotly_white',
#         height=500  # taller figure
#     )
    
#     st.plotly_chart(fig2, use_container_width=True)


# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler, PowerTransformer
# from sklearn.impute import SimpleImputer
# import tensorflow as tf

# import streamlit as st
# import pandas as pd
# import numpy as np
# import tensorflow as tf
# from sklearn.preprocessing import MinMaxScaler
# import joblib

# # --------------------------
# # Load model
# # --------------------------
# model = tf.keras.models.load_model("multivariate_lstm_model.keras")

# # --------------------------
# # Load pre-scaled test sets
# # --------------------------
# X_test = np.loadtxt("X_test.csv", delimiter=",")
# y_test = np.loadtxt("y_test.csv", delimiter=",")

# # Reshape X_test into [samples, timesteps, features]
# timesteps = 20  # must match what you used in training
# n_features = X_test.shape[1] // timesteps
# X_test = X_test.reshape(-1, timesteps, n_features)

# # --------------------------
# # Load target scaler
# # --------------------------
# # If you saved it during training:
# # target_scaler = joblib.load("target_scaler.pkl")

# # Otherwise, refit using original training data (Open & Close only)
# train_df = pd.read_csv("train_data.csv")   # your original training file
# target_scaler = MinMaxScaler(feature_range=(0, 1))
# target_scaler.fit(train_df[['Open', 'Close']].to_numpy())

# # --------------------------
# # Button-triggered prediction
# # --------------------------
# if st.button("▶ Start 10-Day Prediction Simulation"):
#     st.subheader("Predicted vs Previous Day Actual (10-Day Simulation)")

#     X_input = X_test[-1].copy()   # start from last test sequence
#     preds = []
#     prev_actuals = []

#     for day in range(10):
#         X_input_reshaped = X_input.reshape(1, timesteps, n_features)
#         pred_scaled = model.predict(X_input_reshaped, verbose=0)

#         # Inverse transform prediction
#         pred = target_scaler.inverse_transform(pred_scaled)
#         preds.append(pred[0])

#         # Previous actual (scaled → inverse transform)
#         prev_actual_scaled = X_input[-1, :2].reshape(1, -1)
#         prev_actual = target_scaler.inverse_transform(prev_actual_scaled)
#         prev_actuals.append(prev_actual[0])

#         # Update rolling window with predicted Open & Close
#         new_row = X_input[-1].copy()
#         new_row[0], new_row[1] = pred_scaled[0][0], pred_scaled[0][1]  # keep scaled for model
#         X_input = np.vstack([X_input[1:], new_row])

#     # --------------------------
#     # Build results DataFrame
#     # --------------------------
#     pred_df = pd.DataFrame(preds, columns=['Predicted Open', 'Predicted Close'])
#     pred_df['Prev Day Open'] = [pa[0] for pa in prev_actuals]
#     pred_df['Prev Day Close'] = [pa[1] for pa in prev_actuals]
#     pred_df.index = [f"Day {i+1}" for i in range(10)]

#     st.dataframe(pred_df)
