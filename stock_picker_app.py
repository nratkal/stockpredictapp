import streamlit as st
import yfinance as yf

st.title("YFinance test")

ticker = st.text_input("Enter ticker", "MSFT")
if ticker:
    df = yf.download(ticker, period="1mo")
    st.write(df)
