import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ---- Helper functions ----

def fetch_data(ticker, period="1y", interval="1d"):
    """Fetch historical price data from yfinance."""
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty:
            return None
        return df
    except Exception as e:
        return None

def prepare_features(df):
    """Create features for model: returns and moving averages."""
    df = df.copy()
    df['Return'] = df['Adj Close'].pct_change()
    df['MA5'] = df['Adj Close'].rolling(window=5).mean()
    df['MA10'] = df['Adj Close'].rolling(window=10).mean()
    df['MA20'] = df['Adj Close'].rolling(window=20).mean()
    df = df.dropna()
    return df

def create_labels(df, threshold=0.01):
    """Label next-day price movement as buy(1), hold(0), sell(-1) based on threshold."""
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
    """Train a simple RandomForestClassifier on combined ticker data."""
    data_list = []
    for ticker in tickers:
        df = fetch_data(ticker)
        if df is None:
            continue
        df = prepare_features(df)
        df = create_labels(df)
        features = df[['Return', 'MA5', 'MA10', 'MA20']]
        labels = df['Label']
        # Add ticker as a feature if you want, but here we combine all tickers
        data_list.append((features, labels))
    if not data_list:
        return None

    X = pd.concat([x[0] for x in data_list])
    y = pd.concat([x[1] for x in data_list])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    return model, acc

def predict_action(ticker, model):
    """Predict buy/hold/sell for the latest data of a ticker."""
    df = fetch_data(ticker, period="1mo")
    if df is None or df.empty:
        return "No data"
    df = prepare_features(df)
    latest_features = df[['Return', 'MA5', 'MA10', 'MA20']].iloc[-1:].values
    pred = model.predict(latest_features)[0]
    if pred == 1:
        return "buy"
    elif pred == 0:
        return "hold"
    elif pred == -1:
        return "sell"
    else:
        return "hold"

# ---- Streamlit App ----

st.title("📈 AI Stock Picker (Tech Stocks)")

DEFAULT_TECH_TICKERS = ["MSFT", "NVDA", "AAPL", "GOOGL", "AMZN", "AVGO", "PLTR", "TSM"]

st.write("Training AI model on tech stocks. This might take a moment...")

model_acc = None
model = None

try:
    model, model_acc = train_model(DEFAULT_TECH_TICKERS)
except Exception as e:
    st.error(f"Error training model: {e}")

if model is None:
    st.error("Model training failed. Please check your tickers or try again later.")
    st.stop()

st.success(f"Model trained with accuracy: {model_acc:.2f}")

# Predict and show buy candidates from default tech list
buy_candidates = []
for ticker in DEFAULT_TECH_TICKERS:
    action = predict_action(ticker, model)
    if action == "buy":
        buy_candidates.append(ticker)

if buy_candidates:
    st.subheader("Tech Stocks AI Recommends to BUY Right Now:")
    st.write(", ".join(buy_candidates))
else:
    st.info("No tech stocks from the default list are recommended to buy right now.")

# Allow user to enter custom tickers
st.subheader("Check Your Own Tickers")
custom_input = st.text_input("Enter ticker symbols separated by commas (e.g., TSLA, IBM):")

if custom_input:
    user_tickers = [t.strip().upper() for t in custom_input.split(",")]
    for ticker in user_tickers:
        try:
            action = predict_action(ticker, model)
            st.write(f"**{ticker}**: {action}")
        except Exception as e:
            st.write(f"**{ticker}**: Error - {e}")
