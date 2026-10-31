# TCS Stock Model Interface

This folder contains a Streamlit interface for the supplied TCS dataset and trained Linear Regression model.

## Files
- `app.py` — Streamlit web interface
- `stock_model.pkl` — supplied trained model (renamed from the uploaded file)
- `TCS.csv` — supplied dataset
- `requirements.txt` — Python package list

## Run the project
1. Install Python 3.10 or newer on a laptop/desktop.
2. Open a terminal in this folder.
3. Run: `pip install -r requirements.txt`
4. Run: `streamlit run app.py`
5. Open the local URL shown in the terminal.

## Important model limitation
The model file contains a fitted scikit-learn LinearRegression object with 16 named input features. The original training notebook/code and target-variable description were not supplied. As a result, this project displays the model's raw numeric output and does **not** claim that it is the next day's closing price. Ask the model provider to confirm the target variable, prediction horizon, and units before presenting the output as a stock-price forecast.

The model was serialized with scikit-learn 1.9.1, so the requirements file pins that version for compatibility.
