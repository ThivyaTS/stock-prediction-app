import streamlit as st
import os
from google import genai  # make sure google-genai>=0.11.0 is in requirements.txt


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
from tensorflow.keras.models import load_model
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Stock Prediction Dashboard", layout="wide")

st.title("📊 Stock Prediction & SHAP Explanation Dashboard")

# -----------------------------
# 1. Load test data & model
# -----------------------------
TEST_FILE_PATH = "X_test.csv"  # your test file path in Streamlit directory
MODEL_PATH = "multivariate_lstm_model.keras"      # your trained model path

@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

@st.cache_resource
def load_trained_model(path):
    return load_model(path)

test_df = load_data(TEST_FILE_PATH)
model = load_trained_model(MODEL_PATH)

feature_cols = [col for col in test_df.columns if col not in ['Close', 'Open']]
X_test = test_df[feature_cols].values.reshape(test_df.shape[0], 1, len(feature_cols))

# -----------------------------
# 2. Make predictions
# -----------------------------
preds = model.predict(X_test)
results_df = pd.DataFrame({
    "Prev Close": test_df['Close'],
    "Prev Open": test_df['Open'],
    "Pred Close": preds[:, 0],
    "Pred Open": preds[:, 1]
})

st.subheader("Predictions Table")
st.dataframe(results_df)

# -----------------------------
# 3. Generate SHAP summary
# -----------------------------
st.subheader("SHAP Summary Plot")
@st.cache_data
def compute_shap(model, X):
    explainer = shap.KernelExplainer(model.predict, X)
    shap_values = explainer.shap_values(X)
    return shap_values

shap_values = compute_shap(model, X_test)

fig, ax = plt.subplots()
shap.summary_plot(shap_values, X_test, feature_names=feature_cols, show=False)
st.pyplot(fig)

# Store SHAP values for LLM prompts
shap_df = pd.DataFrame(np.array(shap_values).reshape(X_test.shape[0], len(feature_cols)), columns=feature_cols)

# Optional: save SHAP to Excel for download
shap_df.to_excel("shap_summary.xlsx", index=False)
st.download_button("Download SHAP Summary", "shap_summary.xlsx")

# -----------------------------
# 4. LLM topic selection & prompt
# -----------------------------
st.subheader("LLM Explanation")
topic = st.selectbox("Select explanation topic:", ["Predicted Close", "Predicted Open"])

# Generate top features automatically
if topic == "Predicted Close":
    top_features = shap_df.mean().sort_values(ascending=False).head(5)
else:
    top_features = shap_df.mean().sort_values(ascending=False).head(5)  # adjust logic if needed

# Build LLM prompt
prompt = f"Explain how the following features contribute to {topic} prediction:\n"
for feature, value in top_features.items():
    prompt += f"- {feature}: average SHAP value {value:.4f}\n"

# Display prompt (for now) and call your LLM integration
st.text_area("LLM Prompt", prompt, height=200)

# Example placeholder for your LLM integration
# Replace the next lines with your LLM API call
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

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib

# --------------------------
# Load model
# --------------------------
model = tf.keras.models.load_model("multivariate_lstm_model.keras")

# --------------------------
# Load pre-scaled test sets
# --------------------------
X_test = np.loadtxt("X_test.csv", delimiter=",")
y_test = np.loadtxt("y_test.csv", delimiter=",")

# Reshape X_test into [samples, timesteps, features]
timesteps = 20  # must match what you used in training
n_features = X_test.shape[1] // timesteps
X_test = X_test.reshape(-1, timesteps, n_features)

# --------------------------
# Load target scaler
# --------------------------
# If you saved it during training:
# target_scaler = joblib.load("target_scaler.pkl")

# Otherwise, refit using original training data (Open & Close only)
train_df = pd.read_csv("train_data.csv")   # your original training file
target_scaler = MinMaxScaler(feature_range=(0, 1))
target_scaler.fit(train_df[['Open', 'Close']].to_numpy())

# --------------------------
# Button-triggered prediction
# --------------------------
if st.button("▶ Start 10-Day Prediction Simulation"):
    st.subheader("Predicted vs Previous Day Actual (10-Day Simulation)")

    X_input = X_test[-1].copy()   # start from last test sequence
    preds = []
    prev_actuals = []

    for day in range(10):
        X_input_reshaped = X_input.reshape(1, timesteps, n_features)
        pred_scaled = model.predict(X_input_reshaped, verbose=0)

        # Inverse transform prediction
        pred = target_scaler.inverse_transform(pred_scaled)
        preds.append(pred[0])

        # Previous actual (scaled → inverse transform)
        prev_actual_scaled = X_input[-1, :2].reshape(1, -1)
        prev_actual = target_scaler.inverse_transform(prev_actual_scaled)
        prev_actuals.append(prev_actual[0])

        # Update rolling window with predicted Open & Close
        new_row = X_input[-1].copy()
        new_row[0], new_row[1] = pred_scaled[0][0], pred_scaled[0][1]  # keep scaled for model
        X_input = np.vstack([X_input[1:], new_row])

    # --------------------------
    # Build results DataFrame
    # --------------------------
    pred_df = pd.DataFrame(preds, columns=['Predicted Open', 'Predicted Close'])
    pred_df['Prev Day Open'] = [pa[0] for pa in prev_actuals]
    pred_df['Prev Day Close'] = [pa[1] for pa in prev_actuals]
    pred_df.index = [f"Day {i+1}" for i in range(10)]

    st.dataframe(pred_df)
