import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler 
import os
from google import genai  # make sure google-genai>=0.11.0 is in requirements.txt

import plotly.graph_objects as go

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

st.set_page_config(
    page_title="Stock Dashboard",
    layout="wide"  # makes the page full-width
)

st.title("Stock Price Visualization")

# Load dataset
@st.cache_data  # caches the dataset for faster reload
def load_data():
    data = pd.read_csv("dataFrame no last 5 rows.csv")
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
    st.subheader(f"{selected_ticker} Close Prices")
    
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=ticker_data['Date'],
        y=ticker_data['Close'],
        mode='lines',
        name='Close'
    ))

    max_date = ticker_data['Date'].max()

    fig1.update_layout(
        title="Close Prices",
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









# -----------------------------
# 6. LLM topic selection & prompt
# -----------------------------

# --- Get API key from environment variable or Streamlit secrets ---
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    st.error("GEMINI_API_KEY is missing! Add it in Streamlit Secrets.")
    st.stop()

# Initialize Gemini client
client = genai.Client(api_key=API_KEY)

st.subheader("LLM Explanation")
topic = st.selectbox("Select explanation topic:", ["Predicted Close", "Predicted Open"])

prompt = f"Explain how the following features contribute to {topic} prediction:\n"

st.text_area("LLM Prompt", prompt, height=200)

# Placeholder for LLM integration
# llm_response = your_llm_api_call(prompt)
# st.subheader("LLM Explanation")
# st.write(llm_response)

# Placeholder for actual LLM integration
# llm_response = your_llm_api_call(prompt)
# st.subheader("LLM Explanation")
# st.write(llm_response)


# User input: dropdown or text
# topic = st.selectbox("Choose a topic:", ["Artificial Intelligence", "Stock Market", "Crypto", "Finance"])
# user_prompt = st.text_area("Or enter your own prompt:")

# if st.button("Generate Content"):
#     # Construct the prompt
#     prompt_text = user_prompt if user_prompt else f"Explain {topic} in a few words"

#     try:
#         # Generate content using Gemini 2.5 Flash
#         response = client.models.generate_content(
#             model="gemini-2.5-flash",
#             contents=prompt_text
#         )

#         # Display the result
#         st.subheader("Generated Content")
#         st.write(response.text)

#     except Exception as e:
#         st.error(f"Error generating content: {e}")
