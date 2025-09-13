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
import base64
import mimetypes
import os
from google import genai
from google.genai import types
from sklearn.preprocessing import MinMaxScaler 
import logging
import warnings

def set_fixed_background(image_path):
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)





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
    layout="wide",  # makes the page full-width
)
set_fixed_background("bg_thivya_web.jpg")

# -------------------------
# Title with small Apple image
# -------------------------
# -------------------------
# Centered Apple logo + Title using columns
# -------------------------
col1, col2, col3 = st.columns([2.9, 2, 2])  # middle col is wider
with col2:
    st.image("apple_image.png", width=200)  # Apple logo

col1, col2, col3 = st.columns([2.9, 4, 2])  # middle col is wider
with col2:
    st.markdown(
      "<h1 style='font-size:36px; margin-top:10px;'>📊 Stock Price Visualization</h1>",
      unsafe_allow_html=True
    )

st.markdown("---")  # optional horizontal divider

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
    st.subheader("Apple Inc. (AAPL) Stock Close Price Trend")

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=data['Date'],
        y=data['Close'],
        mode='lines',
        name='Close'
    ))

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
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=False),
            type="date",
        ),
        template='plotly_white',
        height=500,
        plot_bgcolor='rgba(10, 10, 10,0.7)',   # chart area: black, 70% opacity
        paper_bgcolor='rgba(0,0,0,0.0)'   # outside area: black, 70% opacity

    )
    st.plotly_chart(fig1, use_container_width=True)


# -------------------------
# Right Column: Volume
# -------------------------
with col2:
    st.subheader("Volume")
    
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=data['Date'],
        y=data['Volume'],
        name='Volume',
        marker_color='orange'
    ))

    fig2.update_layout(
        title="Trading Volume",
        xaxis_title='Date',
        yaxis_title='Volume',
        xaxis=dict(rangeslider=dict(visible=True)),
        template='plotly_white',
        height=500,
        plot_bgcolor='rgba(0,0,0,7)',   # transparent inside chart
        paper_bgcolor='rgba(0,0,0,0)'   # transparent outside chart
    )
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")  # optional horizontal divider

# --- Load financial data ---
@st.cache_data
def load_financial_data():
    df = pd.read_csv("aapl_fin.csv")
    df = df.dropna(how='all')  # Drop empty rows
    df.columns = df.columns.str.strip()
    df['year'] = pd.to_datetime(df['year'], errors='coerce').dt.year
    df = df.dropna(subset=['year'])
    df['year'] = df['year'].astype(int).astype(str)
    df = df.sort_values('year')
    return df

fin_data = load_financial_data()

# ----------------------------
# Column layout: plot | table
# ----------------------------

# --- Load financial data ---
@st.cache_data
def load_financial_data():
    df = pd.read_csv("aapl_fin.csv")
    df = df.dropna(how='all')  # Drop empty rows
    df.columns = df.columns.str.strip()
    df['year'] = pd.to_datetime(df['year'], errors='coerce').dt.year
    df = df.dropna(subset=['year'])
    df['year'] = df['year'].astype(int).astype(str)
    df = df.sort_values('year')
    return df

fin_data = load_financial_data()

# ----------------------------
# Format numbers to billions (e.g., 80000000000 -> 80B)
# ----------------------------
def format_billions(value):
    if pd.isna(value):
        return "-"
    return f"{value / 1e9:.0f}B"

# Select only columns to display in the table
display_cols = ["year", "EPS", "Net profit margin", "EBITDA", "Operating expense"]
table_data = fin_data[display_cols].copy()

# Format the large financial numbers
table_data["EBITDA"] = table_data["EBITDA"].apply(format_billions)
table_data["Operating expense"] = table_data["Operating expense"].apply(format_billions)

table_data["EPS"] = table_data["EPS"].round(2)
table_data["Net profit margin"] = table_data["Net profit margin"].round(2)

# Reset index to remove index column in table
table_data = table_data.reset_index(drop=True)

# ----------------------------
# Layout: Plot + Table Side-by-Side
# ----------------------------
col_chart, col_table = st.columns([3, 2])

# 📊 Plot (unchanged)
with col_chart:
    st.subheader("Revenue & Net Income")
    fig = go.Figure()

    for metric in ["Revenue", "Net income"]:
        if metric in fin_data.columns:
            fig.add_trace(go.Bar(
                x=fin_data['year'],
                y=fin_data[metric],
                name=metric
            ))

    fig.update_layout(
        barmode='group',
        xaxis_title="Fiscal Year",
        yaxis_title="Amount (raw scale)",
        template="plotly_white",
        height=500,
        plot_bgcolor='rgba(0,0,0,0.7)',   # chart area: black, 70% opacity
        paper_bgcolor='rgba(0,0,0,0.0)',  # outside area: black, 70% opacity
        margin=dict(l=20, r=20, t=50, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)

# 📋 Table with formatted billions
with col_table:
    st.subheader("Other Metrics")
    st.table(table_data)



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

#----------------------------------------------------------
# ==============================
# Step-by-Step Prediction Section
# ==============================

# Initialize storage for predictions if not already present
if "predictions" not in st.session_state:
    st.session_state.predictions = dataFrame.copy()

#new
if "predicted_rows" not in st.session_state:
    st.session_state.predicted_rows = pd.DataFrame(columns=["Date", "Previous Close", "Predicted Close"])


# Button for next prediction
if st.button("🔮 Predict Next Day Close Price"):
    
    # Take latest window rows
    latest_data = st.session_state.predictions[-window:].copy()
    non_numeric_cols = latest_data.select_dtypes(exclude=np.number).columns
    latest_data_numeric = latest_data.drop(columns=non_numeric_cols)

    imputer = SimpleImputer()
    latest_scaled = pd.DataFrame(
        imputer.fit_transform(latest_data_numeric),
        columns=latest_data_numeric.columns
    )

    latest_scaled = pd.DataFrame(
        feature_scaler.transform(latest_scaled),
        columns=latest_scaled.columns
    )

    X_input = latest_scaled.values.reshape(1, window, latest_scaled.shape[1])

    # Predict next day Close
    y_pred_scaled = model.predict(X_input)
    y_pred = target_scaler.inverse_transform(y_pred_scaled)
    predicted_close = y_pred[0][0]

    # ===============================
    # Add Prediction to Session State
    # ===============================
    last_date = st.session_state.predictions.index[-1]
    pred_date = last_date + timedelta(days=1)
    
    # Get previous close value
    previous_close = st.session_state.predictions.loc[last_date, "Close"]
    
    # Add predicted close value
    st.session_state.predictions.loc[pred_date, "Close"] = predicted_close
    
    # Show success message
    st.success(
        f"Predicted Close for {pred_date.date()}: **{predicted_close:.2f}**"
    )
    
    # ==================================
    # Store Predicted Row (Avoid Duplicates)
    # ==================================
    if pred_date.strftime("%Y-%m-%d") not in st.session_state.predicted_rows["Date"].values:
        new_row = pd.DataFrame([{
            "Date": pred_date.strftime("%Y-%m-%d"),
            "Previous Close": round(previous_close, 2),
            "Predicted Close": round(predicted_close, 2)
        }])
        
        st.session_state.predicted_rows = pd.concat(
            [st.session_state.predicted_rows, new_row],
            ignore_index=True
        )

        
    # Load the second CSV file
    df_csv = pd.read_csv("last_5_rows.csv")   # replace with actual path
    df_csv['Date'] = pd.to_datetime(df_csv['Date'])  # ensure Date is datetime
    df_csv.set_index('Date', inplace=True)
    
    # Align CSV data to the same date range as predictions
    common_range = st.session_state.predictions.index[-(window+10):]
    df_csv = df_csv.loc[df_csv.index.isin(common_range)]
    # ===============================
    # Plot Updated Figure
    # ===============================
    fig = go.Figure()
    
    # Plot last (window+10) values for clarity
    fig.add_trace(go.Scatter(
        x=st.session_state.predictions.index[-(window+10):],
        y=st.session_state.predictions['Close'].iloc[-(window+10):],
        mode='lines+markers',
        name='Close (Actual + Predicted)'
    ))
        # Plot the second series (CSV)
    fig.add_trace(go.Scatter(
        x=df_csv.index,
        y=df_csv['Close'],
        mode='markers',
        name='Close (CSV File)'
    ))
    # Layout settings
    fig.update_layout(
        title='Latest Close Price with Predictions',
        xaxis_title='Date',
        yaxis_title='Close Price',
        plot_bgcolor='rgba(0,0,0,0.7)',   # chart area background
        paper_bgcolor='rgba(0,0,0,0.0)',  # outside area background
        template='plotly_white'
    )
    
    # Display chart
    st.plotly_chart(fig, use_container_width=True)



    # -----------------------------
    # Show table of predicted rows
    # -----------------------------
    st.subheader("📅 Prediction Summary Table")

    if not st.session_state.predicted_rows.empty:
        predicted_table = st.session_state.predicted_rows.copy()
        predicted_table["Date"] = pd.to_datetime(predicted_table["Date"]).dt.strftime('%Y-%m-%d')

        st.dataframe(predicted_table, use_container_width=True, hide_index=True)
    else:
        st.info("Click the '🔮 Predict Next Step' button to generate predictions.")


    # -----------------------------
    # SHAP explainer for LSTM
    # -----------------------------
    timesteps = X_input.shape[1]
    features = X_input.shape[2]
    X_train = np.load("X_train.npy")
    # Wrapper to let SHAP work with LSTM
    def model_predict_wrapper(X_flat):
        X_3d = X_flat.reshape(X_flat.shape[0], timesteps, features)
        return model.predict(X_3d)
    
    # -----------------------------
    # Background dataset: 20 sequences from training set
    # -----------------------------
    import numpy as np
    num_bg = min(30, X_train.shape[0])
    idx = np.random.choice(X_train.shape[0], num_bg, replace=False)
    background = X_train[idx]  # shape: (num_bg, timesteps, features)
    background_flat = background.reshape(background.shape[0], -1)
    
    # -----------------------------
    # Initialize KernelExplainer
    # -----------------------------
    explainer = shap.KernelExplainer(model_predict_wrapper, background_flat)
    
    # -----------------------------
    # Sample to explain: latest row
    # -----------------------------
    X_sample_flat = X_input.reshape(X_input.shape[0], -1)  # (1, timesteps*features)
    
    # Compute SHAP values
    shap_values = explainer.shap_values(X_sample_flat)
    
    # -----------------------------
    # Aggregate SHAP over timesteps
    # -----------------------------
    sv_3d = shap_values[0].reshape(X_input.shape[0], timesteps, features)
    sv_sum = np.sum(sv_3d, axis=1)  # (samples, features)
    
    # Average input across timesteps to match shape
    X_agg = np.mean(X_input, axis=1)  # (samples, features)
    X_agg_original = feature_scaler.inverse_transform(X_agg)
    
    # -----------------------------
    # Create summary for LLM (with original feature values)
    # -----------------------------
    feature_summary = {}
    for i, feature in enumerate(latest_scaled.columns):
        feature_summary[feature] = {
            "value": float(X_agg_original[0, i]),       # original scale
            "shap_importance": float(sv_sum[0, i])     # SHAP still based on scaled input
        }
    
    # -----------------------------
    # Convert JSON summary to prompt text
    # -----------------------------
    instruction = f""" For {pred_date} date. DO NOT SHOW NUMBERS. Explain how each today's stock price feature influenced tommorow’s stock price prediction. Use the SHAP importance values to describe whether a feature pushed the predicted price higher or lower 
    compared to the average. Do not mention technical terms like SHAP and its VALUES or the model — instead, explain in plain language. Focus on the relative contribution of each feature, using everyday examples 
    (e.g., 'slightly increased', 'pushed down strongly'). Summarize the most important drivers, and highlight why today’s prediction looks the way it does. Avoid showing raw numbers — 
    .SUMMMARIZE ALL OF THEM. Give little not too much financial literacy on the summarization.It is a MUST to make use of this financial data in {fin_data}.
    IT IS ALSO A MUST TO MENTION that this PREDICTION is purely on HISTORICAL DATA and does not reflect real time trading decisions. Purely for financial literacy purposes.Avoid using special characters like (*?!$)"\n"\n"""
    
    prompt_lines = [instruction]  # start with instruction
    for feature, values in feature_summary.items():
        line = f"- {feature}: value = {values['value']:.2f}, SHAP importance = {values['shap_importance']:.4f}"
        prompt_lines.append(line)
    
    prompt_text = "\n".join(prompt_lines)
    
    def save_binary_file(file_name, data):
        f = open(file_name, "wb")
        f.write(data)
        f.close()
        st.success(f"File saved to: {file_name}")
    
    def generate(user_input):
        client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY"),
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
        generated_text = ""  # <--- Initialize an empty string to collect text chunks
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
                 generated_text += chunk.text + " "  # <--- Accumulate text chunks
        return generated_text.strip()  # Return full concatenated text
      
      # Streamlit UI
      # Streamlit UI
    st.title("Google GenAI Explanation")

    prompt_text = prompt_text.strip()

    final_explanation = generate(prompt_text)  # This must return string

    if final_explanation:
        with st.container():
            st.markdown(
                f"""
                <div style="background-color:#2c5073; padding: 20px; border-radius: 10px; border: 1px solid #ddd;">
                    <p style="font-size: 16px; line-height: 1.6; color: #black;">{final_explanation}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.warning("No explanation was generated.")




    
    
    











    
