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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import pickle
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Stock Prediction Dashboard", layout="wide")
st.title("📊 Stock Prediction & SHAP Explanation Dashboard")

# -----------------------------
# 1. Load test data, scaler & model
# -----------------------------
X_TEST_PATH = "X_test_scaled.npy"
Y_TEST_PATH = "y_test_scaled.npy"
MODEL_PATH = "trained_model_aapl.h5"

@st.cache_data
def load_npy(path):
    return np.load(path)

@st.cache_resource
def load_trained_model(path):
    return load_model(path)

import pickle
with open("target_scaler.pkl", "rb") as f:
    target_scaler = pickle.load(f)
    

X_test_scaled = load_npy(X_TEST_PATH)
y_test_scaled = load_npy(Y_TEST_PATH)
model = load_trained_model(MODEL_PATH)
# target_scaler = MinMaxScaler(feature_range=(0, 1))

feature_cols = [f"Feature_{i}" for i in range(X_test_scaled.shape[2])]

# -----------------------------
# 2. Make predictions
# -----------------------------
preds_scaled = model.predict(X_test_scaled)

# -----------------------------
# 3. Inverse transform scaled values
# -----------------------------
# Reshape if needed (2D: samples x 2)
preds_reshaped = preds_scaled.reshape(-1, 2)
y_test_reshaped = y_test_scaled.reshape(-1, 2)

# Inverse transform
preds_real = target_scaler.inverse_transform(preds_reshaped)
y_test_real = target_scaler.inverse_transform(y_test_reshaped)

# -----------------------------
# 4. Create predictions table (t-1 alignment)
# -----------------------------
prev_close = y_test_real[:-1, 0]  # t-1 Close
prev_open  = y_test_real[:-1, 1]  # t-1 Open
pred_close = preds_real[1:, 0]    # predicted Close at t
pred_open  = preds_real[1:, 1]    # predicted Open at t

results_df = pd.DataFrame({
    "Prev Close": prev_close,
    "Prev Open": prev_open,
    "Pred Close": pred_close,
    "Pred Open": pred_open
})

st.subheader("Predictions Table (t-1 vs t, real prices)")
st.dataframe(results_df)

# -----------------------------
# 5. Generate SHAP summary
# -----------------------------
st.subheader("SHAP Summary Plot")

@st.cache_data
def compute_shap(model, X):
    explainer = shap.KernelExplainer(model.predict, X)
    shap_values = explainer.shap_values(X)
    return shap_values

shap_values = compute_shap(model, X_test_scaled)

fig, ax = plt.subplots()
shap.summary_plot(shap_values, X_test_scaled, feature_names=feature_cols, show=False)
st.pyplot(fig)

# Store SHAP values
shap_df = pd.DataFrame(np.array(shap_values).reshape(X_test_scaled.shape[0], X_test_scaled.shape[2]), columns=feature_cols)

# Optional download
shap_df.to_excel("shap_summary.xlsx", index=False)
st.download_button("Download SHAP Summary", "shap_summary.xlsx")

# -----------------------------
# 6. LLM topic selection & prompt
# -----------------------------
st.subheader("LLM Explanation")
topic = st.selectbox("Select explanation topic:", ["Predicted Close", "Predicted Open"])

top_features = shap_df.mean().sort_values(ascending=False).head(5)

prompt = f"Explain how the following features contribute to {topic} prediction:\n"
for feature, value in top_features.items():
    prompt += f"- {feature}: average SHAP value {value:.4f}\n"

st.text_area("LLM Prompt", prompt, height=200)

# Placeholder for LLM integration
# llm_response = your_llm_api_call(prompt)
# st.subheader("LLM Explanation")
# st.write(llm_response)

# Placeholder for actual LLM integration
# llm_response = your_llm_api_call(prompt)
# st.subheader("LLM Explanation")
# st.write(llm_response)


st.title("Gemini 2.5 Flash Content Generator")

# --- Get API key from environment variable or Streamlit secrets ---
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    st.error("GEMINI_API_KEY is missing! Add it in Streamlit Secrets.")
    st.stop()

# Initialize Gemini client
client = genai.Client(api_key=API_KEY)

# User input: dropdown or text
topic = st.selectbox("Choose a topic:", ["Artificial Intelligence", "Stock Market", "Crypto", "Finance"])
user_prompt = st.text_area("Or enter your own prompt:")

if st.button("Generate Content"):
    # Construct the prompt
    prompt_text = user_prompt if user_prompt else f"Explain {topic} in a few words"

    try:
        # Generate content using Gemini 2.5 Flash
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt_text
        )

        # Display the result
        st.subheader("Generated Content")
        st.write(response.text)

    except Exception as e:
        st.error(f"Error generating content: {e}")
