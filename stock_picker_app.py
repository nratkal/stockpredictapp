import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.title("📈 AI Stock Picker (Price-Based)")

# Default tech stocks
DEFAULT_TECH_TICKERS = ["MSFT", "NVDA", "AAPL", "GOOGL", "AMZN", "AVGO", "PLTR", "TSM"]

def fetch_data(ticker, period="1y", interval="1d"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty or 'Adj Close' not in df.columns:
            return None
        return df
    except Exception:
        return None

def prepare_features(df):
    df = df.copy()
    df['Return'] = df['Adj Close'].pct_change()
    df['MA5'] = df['Adj Close'].rolling(window=5).mean()
    df['MA10'] = df['Adj Close'].rolling(window=10).mean()
    df['MA20'] = df['Adj Close'].rolling(window=20).mean()
    df = df.dropna()
    return df

def create_labels(df, threshold=0.01):
    df = df.copy()
    df['Future_Return'] = df['Adj Close'].pct_change().shift(-1)
    df = df.dropna()
    def label_func(x):
        if x > threshold:
            return 1  # buy
        elif x < -threshold:
            return -1  # sell
        else:
            return 0  # hold
    df['Label'] = df['Future_Return'].apply(label_func)
    return df

def train_model(tickers):
    data_list = []
    for ticker in tickers:
        df = fetch_data(ticker)
        if df is None:
            st.warning(f"No valid data for ticker: {ticker}. Skipping.")
            continue
        df = prepare_features(df)
        df = create_labels(df)
        if df.empty:
            st.warning(f"Insufficient data for ticker: {ticker}. Skipping.")
            continue
        features = df[['Return', 'MA5', 'MA10', 'MA20']]
        labels = df['Label']
        data_list.append((features, labels))
    if not data_list:
        return None, None
    X = pd.concat([x[0] for x in data_list])
    y = pd.concat([x[1] for x in data_list])
    if X.empty or y.empty:
        return None, None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    return model, acc

def predict_action(ticker, model):
    df = fetch_data(ticker, period="1mo")
    if df is None or df.empty or 'Adj Close' not in df.columns:
        return "No data"
    df = prepare_features(df)
    if df.empty:
        return "No data"
    latest_features = df[['Return', 'MA5', 'MA10', 'MA20']].iloc[-1:].values
    pred = model.predict(latest_features)[0]
    return {1: "buy", 0: "hold", -1: "sell"}.get(pred, "hold")

st.write("Training AI model on default tech stocks (this may take a moment)...")

model, acc = train_model(DEFAULT_TECH_TICKERS)

if model is None:
    st.error("Model training failed. Please check your tickers or try again later.")
    st.stop()
else:
    st.success(f"Model trained successfully with accuracy: {acc:.2f}")

# Show AI buy recommendations from default tech list
buy_candidates = [t for t in DEFAULT_TECH_TICKERS if predict_action(t, model) == "buy"]
if buy_candidates:
    st.subheader("Tech Stocks Recommended to BUY Now:")
    st.write(", ".join(buy_candidates))
else:
    st.info("No buy recommendations from the default tech stocks at the moment.")

# User input for custom tickers
st.subheader("Check your own tickers")
user_input = st.text_input("Enter tickers separated by commas (e.g. TSLA, IBM):")

if user_input:
    user_tickers = [t.strip().upper() for t in user_input.split(",")]
    st.write("Predictions:")
    for t in user_tickers:
        action = predict_action(t, model)
        st.write(f"**{t}:** {action}")
