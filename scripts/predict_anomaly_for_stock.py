import sys
import os
import pandas as pd
import joblib
from scraper import fetch_stock_data
from engineer_features import compute_technical_features
import yfinance as yf

MODEL_PATH = os.path.join('models', 'isolation_forest.pkl')
RESULTS_PATH = os.path.join('data', 'processed', 'anomaly_results_user_stock.csv')
FEATURES = ['Return', 'LogReturn', 'Volatility', 'RelVolume', 'news_finbert_sentiment', 'reddit_finbert_sentiment']

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/predict_anomaly_for_stock.py <TICKER>")
        sys.exit(1)
    ticker = sys.argv[1]
    print(f"Fetching data for {ticker}...")
    df = fetch_stock_data(ticker, period='1y', interval='1d')
    if df.empty:
        print(f"No data found for ticker {ticker}.")
        sys.exit(1)
    df['Ticker'] = ticker
    if 'Date' not in df.columns:
        df.rename(columns={'index': 'Date'}, inplace=True)
    df = compute_technical_features(df)
    # Add placeholder sentiment columns if not available
    for col in ['news_finbert_sentiment', 'reddit_finbert_sentiment']:
        if col not in df.columns:
            df[col] = 0  # or np.nan if you want to drop these rows
    df = df.dropna(subset=FEATURES)
    X = df[FEATURES]
    print(f"Loading model from {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)
    df['anomaly'] = model.predict(X)
    df.to_csv(RESULTS_PATH, index=False)
    print(f"Anomaly results saved to {RESULTS_PATH}")
    print(df[['Date', 'Ticker', 'anomaly']].tail())

if __name__ == "__main__":
    main() 