from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="TCS Stock Model Interface",
    page_icon="📈",
    layout="wide",
)

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "stock_model.pkl"
DATA_PATH = ROOT / "TCS.csv"

FEATURES = [
    "Prev Close", "Open Price", "High Price", "Low Price", "Close Price",
    "Total Traded Quantity", "Turnover ₹", "No. of Trades",
    "Deliverable Qty", "% Dly Qt to Traded Qty", "MA_7", "MA_14",
    "High_Low_Spread", "Open_Close_Spread", "Price_Return_%", "Volume_Return_%"
]

st.title("📈 TCS Stock Model Interface")
st.caption("A simple dashboard to explore the supplied TCS dataset and run the supplied machine-learning model.")

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = [str(c).strip() for c in df.columns]

    numeric_cols = [
        "Prev Close", "Open Price", "High Price", "Low Price", "Last Price",
        "Close Price", "Average Price", "Total Traded Quantity", "Turnover ₹",
        "No. of Trades", "Deliverable Qty", "% Dly Qt to Traded Qty"
    ]
    for col in numeric_cols:
        if col in df.columns:
            s = df[col].astype(str).str.replace(",", "", regex=False).str.replace("₹", "", regex=False).str.strip()
            df[col] = pd.to_numeric(s, errors="coerce")

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
        df = df.sort_values("Date").reset_index(drop=True)

    close = df["Close Price"] if "Close Price" in df.columns else pd.Series(dtype=float)
    volume = df["Total Traded Quantity"] if "Total Traded Quantity" in df.columns else pd.Series(dtype=float)
    df["MA_7"] = close.rolling(7, min_periods=1).mean()
    df["MA_14"] = close.rolling(14, min_periods=1).mean()
    df["High_Low_Spread"] = df["High Price"] - df["Low Price"]
    df["Open_Close_Spread"] = df["Open Price"] - df["Close Price"]
    df["Price_Return_%"] = close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0) * 100
    df["Volume_Return_%"] = volume.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0) * 100
    return df

@st.cache_resource
def load_model():
    # The supplied file was serialized with scikit-learn 1.9.1.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return joblib.load(MODEL_PATH)

try:
    data = load_data()
    model = load_model()
except Exception as exc:
    st.error("The app could not load its model or dataset.")
    st.exception(exc)
    st.info("Keep app.py, stock_model.pkl, and TCS.csv together in the same folder.")
    st.stop()

with st.sidebar:
    st.header("Navigation")
    page = st.radio("Choose a page", ["Dashboard", "Run Model", "About"])

if page == "Dashboard":
    st.subheader("Dataset overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(data):,}")
    c2.metric("Columns in CSV", f"{len(pd.read_csv(DATA_PATH).columns):,}")
    if "Close Price" in data and data["Close Price"].notna().any():
        c3.metric("Latest available close", f"₹{data['Close Price'].dropna().iloc[-1]:,.2f}")

    if "Date" in data and data["Date"].notna().any():
        st.write("Date range:", data["Date"].min().strftime("%d %b %Y"), "to", data["Date"].max().strftime("%d %b %Y"))

    st.subheader("Closing price trend")
    if "Date" in data and "Close Price" in data:
        chart_data = data.dropna(subset=["Date", "Close Price"]).set_index("Date")[["Close Price"]]
        st.line_chart(chart_data)
    st.subheader("Dataset preview")
    st.dataframe(data.tail(20), use_container_width=True)

elif page == "Run Model":
    st.subheader("Enter model input values")
    st.warning(
        "Important: the supplied model file does not contain the target/output name or forecasting instructions. "
        "This page displays the model's raw numeric output; it must not be described as a next-day closing-price "
        "forecast until the original model documentation confirms that."
    )

    # Use the latest dataset row as convenient starting values.
    latest = data.dropna(subset=[c for c in FEATURES if c in data.columns]).iloc[-1] if len(data) else None
    defaults = {}
    if latest is not None:
        for f in FEATURES:
            defaults[f] = float(latest[f]) if f in latest and pd.notna(latest[f]) else 0.0

    with st.form("prediction_form"):
        st.write("The fields are prefilled from the latest available dataset row. You can edit them.")
        cols = st.columns(2)
        values = {}
        for i, feature in enumerate(FEATURES):
            with cols[i % 2]:
                values[feature] = st.number_input(
                    feature,
                    value=float(defaults.get(feature, 0.0)),
                    format="%.6f",
                    help="Enter a numeric value in the same units/scale as the model's training data."
                )
        submitted = st.form_submit_button("Run model", type="primary")

    if submitted:
        try:
            input_df = pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES)
            prediction = model.predict(input_df)
            st.success("Model ran successfully.")
            st.metric("Raw model output", f"{float(np.asarray(prediction).ravel()[0]):,.6f}")
            st.caption("This is the model's numeric output. Confirm the target variable with the person who supplied the model before interpreting it as a stock price.")
            with st.expander("View the exact input sent to the model"):
                st.dataframe(input_df, use_container_width=True)
        except Exception as exc:
            st.error("Prediction failed. Please check the entered values and model compatibility.")
            st.exception(exc)

else:
    st.subheader("About this project")
    st.markdown(
        """
        - **Dataset:** TCS.csv
        - **Model:** supplied scikit-learn Linear Regression model
        - **Interface:** Streamlit
        - **Model inputs:** 16 numeric features
        - **Purpose:** run the supplied model through a simple web interface and explore the dataset.
        """
    )
    st.info(
        "The original training notebook/code was not supplied. Therefore, the exact target variable, "
        "train/test split, evaluation scores, and intended prediction horizon cannot be verified from the model file alone."
    )
    st.write("Expected model input feature names:")
    st.code("\\n".join(FEATURES))

st.divider()
st.caption("Educational project interface. This is not financial advice or a recommendation to buy or sell shares.")
