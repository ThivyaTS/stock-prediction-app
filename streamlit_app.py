import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import timedelta
from sklearn.impute import SimpleImputer
import shap
from tensorflow import keras
import plotly.graph_objects as go
import joblib

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler 
import os
from google import genai
  # make sure google-genai>=0.11.0 is in requirements.txt

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
    data = pd.read_csv("dataFrame no last 5 rows (1).csv")
    return data

data = load_data()
data['Date'] = pd.to_datetime(data['Date'])

# -------------------------
# Create side-by-side columns
# -------------------------
col1, col2 = st.columns([4, 2])  # left column bigger

# -------------------------
# Left Column: Open & Close Prices
# -------------------------
with col1:
    st.subheader("AAPL Close Prices")
    
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=data['Date'],
        y=data['Close'],
        mode='lines',
        name='Close'
    ))

    # max_date = data['Date'].max()

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
        # range=[data['Date'].min(), max_date]  # show only dataset range
    ),

    template='plotly_white',
    height=500 
) 
st.plotly_chart(fig1, use_container_width=True)

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
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import joblib
from tensorflow import keras
import plotly.graph_objects as go

# -----------------------------
# Load saved model and scalers
# -----------------------------
model = keras.models.load_model("multivariate_lstm_model_aapl.keras")
feature_scaler = joblib.load("feature_scaler_aapl_latest.save")
target_scaler = joblib.load("target_scaler_aapl_latest.save")

# -----------------------------
# Load your dataset
# -----------------------------
dataFrame = pd.read_csv("dataFrame no last 5 rows (1).csv")
dataFrame['Date'] = pd.to_datetime(dataFrame['Date'])
dataFrame.set_index('Date', inplace=True)

# -----------------------------
# Select latest 20 rows
# -----------------------------
window = 20
latest_data = dataFrame[-window:].copy()

# -----------------------------
# Drop non-numeric columns (e.g., ticker or strings)
# -----------------------------
non_numeric_cols = latest_data.select_dtypes(exclude=np.number).columns
latest_data_numeric = latest_data.drop(columns=non_numeric_cols)

# -----------------------------
# Impute missing numeric values
# -----------------------------
imputer = SimpleImputer()
latest_scaled = pd.DataFrame(
    imputer.fit_transform(latest_data_numeric),
    columns=latest_data_numeric.columns
)

# -----------------------------
# Scale features
# -----------------------------
latest_scaled = pd.DataFrame(
    feature_scaler.transform(latest_scaled),
    columns=latest_scaled.columns
)

# -----------------------------
# Reshape for LSTM (samples, timesteps, features)
# -----------------------------
X_input = latest_scaled.values.reshape(1, window, latest_scaled.shape[1])

# -----------------------------
# Predict next day Close
# -----------------------------
y_pred_scaled = model.predict(X_input)
y_pred = target_scaler.inverse_transform(y_pred_scaled)
predicted_close = y_pred[0][0]
from datetime import timedelta

# Ensure last_date is a Timestamp
last_date = pd.to_datetime(dataFrame.index[-1])

# Add 3 days to skip weekend
pred_date = last_date + timedelta(days=3)

# -----------------------------
# Visualization
# -----------------------------
fig = go.Figure()

# Plot last 20 actual Close values
fig.add_trace(go.Scatter(
    x=dataFrame.index[-window:], 
    y=dataFrame['Close'][-window:], 
    mode='lines', 
    name='Actual Close'
))

# Plot predicted next day
# pred_date = dataFrame.index[-1] + pd.Timedelta(days=1)
fig.add_trace(go.Scatter(
    x=[pred_date],
    y=[predicted_close],
    mode='markers',
    name='Predicted Close',
    marker=dict(color='red', size=10)
))

fig.update_layout(
    title='Latest Close Prices + Next Day Prediction',
    xaxis_title='Date',
    yaxis_title='Close Price',
    template='plotly_white'
)

st.plotly_chart(fig, use_container_width=True)
st.write(f"Predicted Close price for {pred_date.date()}: **{predicted_close:.2f}**")

import shap

# -----------------------------
# SHAP Explainer
# -----------------------------
# Use a small background set (latest window)
background = X_input  # already reshaped: (1, window, features)

# KernelExplainer for non-linear models like LSTM
explainer = shap.KernelExplainer(model.predict, background)

# Explain the prediction for X_input
shap_values = explainer.shap_values(X_input)

# Convert to DataFrame for easier visualization
shap_df = pd.DataFrame(
    shap_values[0][0],  # first sample
    columns=latest_scaled.columns
)

st.write("SHAP values for latest input features:")
st.dataframe(shap_df.T)


# Aggregate SHAP values per feature
shap_values_array = shap_values[0].reshape(window, num_features)
feature_importance = np.abs(shap_values_array).sum(axis=0)
feature_names = latest_scaled.columns
shap_summary = dict(zip(feature_names, feature_importance))
shap_summary_sorted = dict(sorted(shap_summary.items(), key=lambda x: x[1], reverse=True))

# # -----------------------------
# # LLM Explanation
# # -----------------------------

shap_text = "\n".join([f"{k}: {v:.4f}" for k, v in shap_summary_sorted.items()])
# prompt_text = f"Explain how the following features contributed to {topic} prediction:\n{shap_text}"
# API_KEY = os.getenv("GEMINI_API_KEY")
# if not API_KEY:
#     st.error("GEMINI_API_KEY is missing! Add it in Streamlit Secrets.")
#     st.stop()

# client = genai.Client(api_key=API_KEY)

# st.subheader("LLM Explanation")
# topic = st.selectbox("Select explanation topic:", ["Predicted Close", "Predicted Open"])



# st.text_area("LLM Prompt", prompt, height=200)

# response = client.models.generate_content(
#     model="gemini-2.5-flash",
#     contents=prompt,
#     max_output_tokens=300
# )

# st.text_area("LLM Response", response.text, height=300)

# -----------------------------
# LLM topic selection & prompt
# -----------------------------

# # --- Get API key from environment variable or Streamlit secrets ---
# API_KEY = os.getenv("GEMINI_API_KEY")
# if not API_KEY:
#     st.error("GEMINI_API_KEY is missing! Add it in Streamlit Secrets.")
#     st.stop()

# # Initialize Gemini client
# client = genai.Client(api_key=API_KEY)

# st.title("Stock Prediction LLM Explainer")

# # User input: dropdown or text
# topic = st.selectbox("Choose a topic:", ["Predicted Close", "Predicted Open", "Artificial Intelligence", "Stock Market", "Crypto", "Finance"])
# user_prompt = st.text_area("Or enter your own prompt:")

# # Placeholder for SHAP text. You must populate this variable with your SHAP explanation text.
# shap_text = "Example SHAP explanation text. You need to replace this with your actual SHAP data."

# if st.button("Generate Content"):
#     # Construct the prompt
#     if user_prompt:
#         prompt_text = user_prompt
#     else:
#         # Check if topic is a stock prediction type and use shap_text
#         if topic in ["Predicted Close", "Predicted Open"]:
#             prompt_text = f"Explain how the following features contributed to {topic} prediction:\n{shap_text}"
#         else:
#             prompt_text = f"Explain {topic}." # A simple prompt for general topics

#     try:
#         # Generate content using Gemini 1.5 Flash
#         response = client.models.generate_content(
#             model="gemini-2.5-flash", contents="Explain how AI works in a few words"
#         )
        
#         # Display the result
#         st.subheader("Generated Content")
#         st.write(response.text)

#     except Exception as e:
#         st.error(f"Error generating content: {e}")

import base64
import mimetypes
import os
import streamlit as st
from google import genai
from google.genai import types


def save_binary_file(file_name, data):
    f = open(file_name, "wb")
    f.write(data)
    f.close()
    st.success(f"File saved to: {file_name}")

def generate(user_input):
    client = genai.Client(
        api_key="AIzaSyAxfhlsbPeJ4ipjp72-0SnJ1G879WqfTs4",
    )

    model = "gemini-2.5-flash"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=user_input),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=0.05,
        response_modalities=[
            "TEXT",
        ],
    )

    file_index = 0
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        if (
            chunk.candidates is None
            or chunk.candidates[0].content is None
            or chunk.candidates[0].content.parts is None
        ):
            continue
        if chunk.candidates[0].content.parts[0].inline_data and chunk.candidates[0].content.parts[0].inline_data.data:
            file_name = f"ENTER_FILE_NAME_{file_index}"
            file_index += 1
            inline_data = chunk.candidates[0].content.parts[0].inline_data
            data_buffer = inline_data.data
            file_extension = mimetypes.guess_extension(inline_data.mime_type)
            save_binary_file(f"{file_name}{file_extension}", data_buffer)
        else:
            st.text(chunk.text)

# Streamlit UI
st.title("Google GenAI Explanation")




# st.write("**What would you like to know about the prediction?**")
# st.text_area("", user_input, height=150)

if st.button("Generate"):
      # Example mapping of prompts to pre-existing SHAP texts
      # Get the corresponding SHAP text
  # User selects a prompt
    prompts = {
    "Prompt 1": f"Explain how the following features contributed to prediction:\n{shap_text}",
    "Prompt 2": "SHAP text for prompt 2...",
    "Prompt 3": "SHAP text for prompt 3...",
    # Add more prompts and corresponding SHAP texts as needed
    }
    selected_prompt = st.selectbox("What would you like to know about the prediction?:", list(prompts.keys()))
    user_input = prompts[selected_prompt]
    if user_input.strip():
      generate(user_input)
    else:
        st.warning("No text available for the selected prompt.")










    

