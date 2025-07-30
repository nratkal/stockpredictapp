{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import streamlit as st\
import yfinance as yf\
import pandas as pd\
import numpy as np\
from datetime import datetime, timedelta\
from nltk.sentiment.vader import SentimentIntensityAnalyzer\
import nltk\
from sklearn.ensemble import RandomForestClassifier\
from sklearn.model_selection import train_test_split\
\
import requests\
from bs4 import BeautifulSoup\
\
# Download VADER lexicon for sentiment\
nltk.download('vader_lexicon')\
\
st.title("\uc0\u55357 \u56522  AI Stock Picker with Price + News Sentiment")\
\
# --- Step 1: Setup stock list ---\
# Feel free to expand this list\
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'INTC', 'AMD', 'NFLX']\
\
st.sidebar.header("Settings")\
selected_tickers = st.sidebar.multiselect("Select tickers to analyze:", TICKERS, default=TICKERS)\
\
# --- Step 2: Fetch price data and compute features ---\
@st.cache_data(ttl=3600)\
def fetch_price_data(ticker):\
    df = yf.download(ticker, period="6mo", interval='1d', progress=False)\
    df.dropna(inplace=True)\
    return df\
\
def calculate_technical_features(df):\
    df['Return'] = df['Close'].pct_change()\
    df['MA10'] = df['Close'].rolling(window=10).mean()\
    df['MA50'] = df['Close'].rolling(window=50).mean()\
    df['Volatility'] = df['Close'].rolling(window=10).std()\
    df['Momentum'] = df['Close'] - df['Close'].shift(7)  # 1 week momentum\
    df.dropna(inplace=True)\
    return df\
\
# --- Step 3: Fetch news headlines from Reddit finance subreddit ---\
def fetch_reddit_headlines(ticker, limit=10):\
    url = f"https://www.reddit.com/r/stocks/search.json?q=\{ticker\}&restrict_sr=1&sort=new"\
    headers = \{'User-Agent': 'Mozilla/5.0'\}\
    try:\
        response = requests.get(url, headers=headers, timeout=10)\
        data = response.json()\
        posts = data['data']['children']\
        headlines = [post['data']['title'] for post in posts[:limit]]\
        return headlines\
    except Exception:\
        return []\
\
# --- Step 4: Sentiment analysis with VADER ---\
sia = SentimentIntensityAnalyzer()\
\
def analyze_sentiment(headlines):\
    if not headlines:\
        return 0.0\
    scores = [sia.polarity_scores(headline)['compound'] for headline in headlines]\
    avg_score = np.mean(scores)\
    return avg_score\
\
# --- Step 5: Build training data and model ---\
@st.cache_data(ttl=86400)\
def train_model(tickers):\
    rows = []\
    for ticker in tickers:\
        df = fetch_price_data(ticker)\
        df = calculate_technical_features(df)\
        # Create target: 1 if Close price 7 days later is higher\
        df['Target'] = (df['Close'].shift(-7) > df['Close']).astype(int)\
        df.dropna(inplace=True)\
        for _, row in df.iterrows():\
            rows.append(\{\
                'Ticker': ticker,\
                'MA10': row['MA10'],\
                'MA50': row['MA50'],\
                'Volatility': row['Volatility'],\
                'Momentum': row['Momentum'],\
                'Target': row['Target']\
            \})\
    data = pd.DataFrame(rows)\
    features = ['MA10', 'MA50', 'Volatility', 'Momentum']\
    X = data[features]\
    y = data['Target']\
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\
    model = RandomForestClassifier(n_estimators=100, random_state=42)\
    model.fit(X_train, y_train)\
    acc = model.score(X_test, y_test)\
    return model, acc\
\
model, accuracy = train_model(TICKERS)\
st.sidebar.write(f"Model trained with accuracy: **\{accuracy:.2%\}**")\
\
# --- Step 6: Predict and rank stocks ---\
results = []\
\
for ticker in selected_tickers:\
    df = fetch_price_data(ticker)\
    df = calculate_technical_features(df)\
    if df.empty:\
        continue\
    latest = df.iloc[-1]\
    features = np.array([latest['MA10'], latest['MA50'], latest['Volatility'], latest['Momentum']]).reshape(1, -1)\
    pred = model.predict(features)[0]\
    prob = model.predict_proba(features)[0][pred]\
    \
    headlines = fetch_reddit_headlines(ticker)\
    sentiment_score = analyze_sentiment(headlines)\
    \
    # Composite score: weight 60% model confidence + 40% sentiment\
    composite_score = 0.6 * prob + 0.4 * ((sentiment_score + 1) / 2)  # normalize sentiment from [-1,1] to [0,1]\
    \
    signal = "Buy" if pred == 1 else "Sell/Hold"\
    results.append(\{\
        'Ticker': ticker,\
        'Signal': signal,\
        'Confidence': prob,\
        'Sentiment': sentiment_score,\
        'Composite': composite_score,\
        'Headline Sample': headlines[:3]\
    \})\
\
# Sort results by composite score descending\
results = sorted(results, key=lambda x: x['Composite'], reverse=True)\
\
# --- Step 7: Show top 3 picks ---\
st.header("Top Stock Picks Based on AI Model + Sentiment")\
\
if results:\
    for i, res in enumerate(results[:3], 1):\
        st.subheader(f"\{i\}. \{res['Ticker']\} - \{res['Signal']\}")\
        st.write(f"Confidence Score: \{res['Confidence']:.2f\}")\
        st.write(f"News Sentiment Score: \{res['Sentiment']:.2f\}")\
        st.write(f"Composite Score: \{res['Composite']:.2f\}")\
        st.write("Recent Headlines:")\
        for h in res['Headline Sample']:\
            st.write(f"- \{h\}")\
        st.markdown("---")\
else:\
    st.write("No stock data to show. Try changing the tickers or check your connection.")\
\
}