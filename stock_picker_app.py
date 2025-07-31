import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import requests

nltk.download('vader_lexicon')

# --- App Title ---
st.title("📈 AI Stock Picker (Price + News Sentiment)")
st.write("Predict short-term stock movements using historical prices, technicals, and Reddit news sentiment.")

# --- Sidebar Settings ---
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'INTC', 'AMD', 'NFLX']
selected_tickers = st.sidebar.multiselect("Select tickers to analyze:", TICKERS, default=TICKERS)

# --- Cache Price Data ---
@st.cache_data(ttl=3600)
def fetch_price_data(ticker):
    df = yf.download(ticker, period="6mo", interval='1d', progress=False)
    df.dropna(inplace=True)
    return df

def calculate_features(df):
    df['Return'] = df['Close'].pct_change()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['Volatility'] = df['Close'].rolling(window=10).std()
    df['Momentum'] = df['Close'] - df['Close'].shift(7)
    df['Target'] = (df['Close'].shift(-7) > df['Close']).astype(int)
    df = df.dropna()
    return df[['MA10', 'MA50', 'Volatility', 'Momentum', 'Target']]

# --- News Fetch (Reddit search API) ---
def fetch_reddit_headlines(ticker, limit=10):
    url = f"https://www.reddit.com/r/stocks/search.json?q={ticker}&restrict_sr=1&sort=new"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        data = r.json()
        return [post['data']['title'] for post in data['data']['children'][:limit]]
    except:
        return []

# --- Sentiment Analysis ---
sia = SentimentIntensityAnalyzer()

def analyze_sentiment(headlines):
    if not headlines:
        return 0.0
    scores = [sia.polarity_scores(h)['compound'] for h in headlines]
    return np.mean(scores)

# --- Train AI Model ---
@st.cache_data(ttl=86400)
def train_model(tickers):
    all_data = []
    for ticker in tickers:
        df = fetch_price_data(ticker)
        df_feat = calculate_features(df)
        df_feat['Ticker'] = ticker
        all_data.append(df_feat)

    full_df = pd.concat(all_data, ignore_index=True)
    full_df.dropna(inplace=True)
    full_df = full_df[np.isfinite(full_df[['MA10', 'MA50', 'Volatility', 'Momentum']]).all(axis=1)]

    X = full_df[['MA10', 'MA50', 'Volatility', 'Momentum']]
    y = full_df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    return model, acc

model, acc = train_model(TICKERS)
st.sidebar.markdown(f"**Model Accuracy:** `{acc:.2%}`")

# --- Evaluate Selected Tickers ---
st.header("📊 Stock Picks & Predictions")
results = []

for ticker in selected_tickers:
    df = fetch_price_data(ticker)
    df_feat = calculate_features(df)
    if df_feat.empty:
        continue

    latest = df_feat.iloc[-1]
    X_live = latest[['MA10', 'MA50', 'Volatility', 'Momentum']].values.reshape(1, -1)
    prediction = model.predict(X_live)[0]
    confidence = model.predict_proba(X_live)[0][prediction]

    headlines = fetch_reddit_headlines(ticker)
    sentiment = analyze_sentiment(headlines)
    composite = 0.6 * confidence + 0.4 * ((sentiment + 1) / 2)

    results.append({
        'Ticker': ticker,
        'Signal': 'Buy' if prediction == 1 else 'Sell/Hold',
        'Confidence': confidence,
        'Sentiment': sentiment,
        'Composite': composite,
        'Headlines': headlines[:3]
    })

# --- Show Results ---
results = sorted(results, key=lambda x: x['Composite'], reverse=True)

if not results:
    st.warning("No results to display. Please select valid tickers.")
else:
    for res in results[:3]:
        st.subheader(f"📈 {res['Ticker']} — {res['Signal']}")
        st.write(f"Confidence: `{res['Confidence']:.2f}`")
        st.write(f"Sentiment: `{res['Sentiment']:.2f}`")
        st.write(f"Composite Score: `{res['Composite']:.2f}`")
        st.write("Sample Headlines:")
        for h in res['Headlines']:
            st.markdown(f"- {h}")
        st.markdown("---")
