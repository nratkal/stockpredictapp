import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import nltk
import requests
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import datetime

# Download VADER lexicon
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

st.set_page_config(page_title="AI Stock Picker", layout="wide")
st.title("📈 AI Stock Picker (Price + News Sentiment)")

# Input: Ticker list
ticker_input = st.text_input("Enter stock tickers separated by commas (e.g., AAPL, TSLA, NVDA):")
TICKERS = [t.strip().upper() for t in ticker_input.split(",") if t.strip() != ""]

# Sample fallback tech tickers if none entered
if not TICKERS:
    TICKERS = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN", "AVGO", "PLTR", "TSM"]
    st.info("No tickers entered. Using default tech tickers.")

@st.cache_data(show_spinner=True)
def fetch_stock_data(ticker):
    try:
        df = yf.download(ticker, period="6mo")
        if df.empty:
            return None
        df["Return"] = df["Adj Close"].pct_change()
        df["Target"] = np.where(df["Return"].shift(-1) > 0, 1, 0)
        return df.dropna()
    except Exception:
        return None

@st.cache_data(show_spinner=True)
def get_news_sentiment(ticker):
    # Use a public news search as a placeholder for real News API
    url = f"https://newsapi.org/v2/everything?q={ticker}&pageSize=5&apiKey=demo"
    try:
        r = requests.get(url)
        headlines = [article["title"] for article in r.json().get("articles", [])]
        sentiments = [sia.polarity_scores(title)['compound'] for title in headlines]
        return np.mean(sentiments) if sentiments else 0
    except:
        return 0

@st.cache_data(show_spinner=True)
def train_model(tickers):
    full_df = []
    for ticker in tickers:
        df = fetch_stock_data(ticker)
        if df is not None and not df.empty:
            df["Ticker"] = ticker
            df["Sentiment"] = get_news_sentiment(ticker)
            full_df.append(df)
        else:
            st.warning(f"No valid data for ticker: {ticker}. Skipping.")

    if not full_df:
        return None, None

    data = pd.concat(full_df)
    X = data[["Return", "Sentiment"]]
    y = data["Target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc

# Train the model
with st.spinner("Training AI model on selected stocks. This might take a moment..."):
    model, acc = train_model(TICKERS)

if model:
    st.success(f"Model trained with accuracy: {acc*100:.2f}%")

    st.subheader("📊 Stock Predictions")
    predictions = {}
    for ticker in TICKERS:
        df = fetch_stock_data(ticker)
        if df is not None and not df.empty:
            latest = df.iloc[-1]
            input_data = [[latest["Return"], get_news_sentiment(ticker)]]
            pred = model.predict(input_data)[0]
            action = "📈 Buy" if pred == 1 else "⚠️ Hold/Sell"
            predictions[ticker] = action
            st.markdown(f"**{ticker}** — {action}")

    buy_recos = [ticker for ticker, action in predictions.items() if "Buy" in action]
    if buy_recos:
        st.subheader("🔥 Suggested Buys (Tech)")
        st.write(", ".join(buy_recos))
    else:
        st.info("No strong buy signals currently. Try again later or adjust tickers.")
else:
    st.error("Model training failed. Please check your tickers or try again later.")
