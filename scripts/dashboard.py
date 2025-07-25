import streamlit as st
import pandas as pd
import os
import joblib
from scraper import fetch_stock_data
from engineer_features import compute_technical_features

PROCESSED_DATA = os.path.join('data', 'processed', 'anomaly_results.csv')
ALL_FEATURES = os.path.join('data', 'processed', 'all_features_bert.csv')
NEWS = os.path.join('data', 'raw', 'news_data.csv')
REDDIT = os.path.join('data', 'raw', 'reddit_data.csv')
MODEL_PATH = os.path.join('models', 'isolation_forest.pkl')
# FEATURES = ['Return', 'LogReturn', 'Volatility', 'RelVolume', 'news_finbert_sentiment', 'reddit_finbert_sentiment']
REQUIRED_FEATURES = ['Return', 'LogReturn', 'Volatility', 'RelVolume', 'news_finbert_sentiment', 'reddit_finbert_sentiment']

st.title("Pump and Dump Anomaly Detector")

# User input for any ticker
user_ticker = st.text_input("Enter any stock ticker (e.g., INFY.NS) to analyze:")
user_df = None
user_error = None
if user_ticker:
    try:
        st.write(f"Fetching and analyzing data for {user_ticker}...")
        df = fetch_stock_data(user_ticker, period='1y', interval='1d')
        if df.empty:
            user_error = f"No data found for ticker {user_ticker}."
        else:
            df['Ticker'] = user_ticker
            if 'Date' not in df.columns:
                df.rename(columns={'index': 'Date'}, inplace=True)
            df = compute_technical_features(df)
            for col in ['news_finbert_sentiment', 'reddit_finbert_sentiment']:
                if col not in df.columns:
                    df[col] = 0
            df = df.dropna(subset=REQUIRED_FEATURES)
            if not df.empty:
                X = df[REQUIRED_FEATURES]
                model = joblib.load(MODEL_PATH)
                df['anomaly'] = model.predict(X)
                user_df = df
            else:
                user_error = f"Not enough data after feature engineering for {user_ticker}."
    except Exception as e:
        user_error = f"Error processing {user_ticker}: {e}"

if user_ticker and user_df is not None:
    st.subheader(f"Anomaly Detection for {user_ticker}")
    st.line_chart(user_df.set_index('Date')[['Close', 'Volume']])
    st.markdown("### Anomaly Flags (Red = Anomaly)")
    st.line_chart(user_df.set_index('Date')['anomaly'])
    st.markdown("### Flagged Anomalies")
    anomalies = user_df[user_df['anomaly'] == -1]
    st.dataframe(anomalies[['Date', 'Close', 'anomaly']])
elif user_ticker and user_error:
    st.error(user_error)
else:
    # Default: show selectbox for existing tickers
    df = pd.read_csv(PROCESSED_DATA)
    features = pd.read_csv(ALL_FEATURES)
    news = pd.read_csv(NEWS)
    reddit = pd.read_csv(REDDIT)
    df['Date'] = pd.to_datetime(df['Date'])
    features['Date'] = pd.to_datetime(features['Date'])
    tickers = df['Ticker'].unique()
    selected = st.selectbox("Select Stock", tickers)
    stock_df = df[df['Ticker'] == selected]
    stock_features = features[features['Ticker'] == selected]
    st.line_chart(stock_df.set_index('Date')[['Close', 'Volume']])
    st.markdown("### Anomaly Flags (Red = Anomaly)")
    st.line_chart(stock_df.set_index('Date')['anomaly'])
    st.markdown("### News & Reddit FinBERT Sentiment")
    st.line_chart(stock_features.set_index('Date')[['news_finbert_sentiment', 'reddit_finbert_sentiment']])
    st.markdown("### Flagged Anomalies with Context")
    anomalies = stock_df[stock_df['anomaly'] == -1]
    for _, row in anomalies.iterrows():
        st.write(f"**Date:** {row['Date'].date()} | **Close:** {row['Close']}")
        news_on_date = news[news['title'].notnull() & (pd.to_datetime('today').date() == row['Date'].date())]
        if not news_on_date.empty:
            st.write("**News Headlines:**")
            for _, nrow in news_on_date.iterrows():
                st.write(f"- {nrow['title']}")
        reddit_on_date = reddit[pd.to_datetime(reddit['created_utc'], unit='s').dt.date == row['Date'].date()]
        if not reddit_on_date.empty:
            st.write("**Reddit Posts:**")
            for _, rrow in reddit_on_date.iterrows():
                st.write(f"- {rrow['title']}")
        st.write("---")

# SHAP explainability (if available)
shap_path = os.path.join('data', 'processed', 'shap_values.csv')
if os.path.exists(shap_path):
    st.markdown("### Model Explainability (SHAP)")
    shap_df = pd.read_csv(shap_path)
    st.dataframe(shap_df)