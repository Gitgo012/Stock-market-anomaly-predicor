import pandas as pd
import numpy as np
import os
import re

# Define paths
RAW_MARKET = os.path.join('data', 'raw', 'market_data.csv')
NEWS_SENTIMENT = os.path.join('data', 'processed', 'news_sentiment_bert.csv')
REDDIT_SENTIMENT = os.path.join('data', 'processed', 'reddit_sentiment_bert.csv')
PROCESSED_DIR = os.path.join('data', 'processed')
PROCESSED_DATA = os.path.join(PROCESSED_DIR, 'all_features_bert.csv')

# Create processed directory if not exists
os.makedirs(PROCESSED_DIR, exist_ok=True)

def compute_technical_features(df):
    df = df.sort_values('Date').copy()
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
    df.dropna(subset=['Close'], inplace=True)
    df['Return'] = df['Close'].pct_change()
    df['LogReturn'] = np.log(df['Close'] / df['Close'].shift(1))
    df['LogReturn'].replace([np.inf, -np.inf], np.nan, inplace=True)
    df['Volatility'] = df['Return'].rolling(window=5).std()
    df['Volume_MA'] = df['Volume'].rolling(window=5).mean()
    df['RelVolume'] = df['Volume'] / df['Volume_MA']
    df.dropna(inplace=True)
    return df

def sentiment_to_score(label):
    label = str(label).lower()
    if label == 'positive': return 1
    if label == 'negative': return -1
    return 0

def extract_ticker_from_title(title, tickers):
    # Try to find a ticker in the title from a list of known tickers
    for ticker in tickers:
        if ticker.lower() in title.lower():
            return ticker.upper()
    return None

def main():
    # === Load and process market data ===
    market = pd.read_csv(RAW_MARKET)
    print("Loaded market data:", market.shape)
    market['Ticker'] = market['Ticker'].str.upper()
    market = market.groupby('Ticker').apply(compute_technical_features).reset_index(drop=True)
    market['Date'] = pd.to_datetime(market['Date']).dt.date
    print("After technical feature engineering:", market.shape)

    # Get list of tickers from market data
    known_tickers = market['Ticker'].unique().tolist()

    # === Process News Sentiment ===
    try:
        news = pd.read_csv(NEWS_SENTIMENT)
        print("Loaded news sentiment:", news.shape)
        if 'date' in news.columns:
            news['date'] = pd.to_datetime(news['date']).dt.date
        else:
            news['date'] = pd.NaT
        # Try to extract ticker from title if not present
        if 'Ticker' not in news.columns:
            news['Ticker'] = news['title'].apply(lambda x: extract_ticker_from_title(str(x), known_tickers))
        news['Ticker'] = news['Ticker'].str.upper()
        news['finbert_score'] = news['finbert_sentiment'].apply(sentiment_to_score)
        # Aggregate by date and ticker
        news_agg = news.groupby(['date', 'Ticker'], dropna=False)['finbert_score'].mean().reset_index().rename(
            columns={'finbert_score': 'news_finbert_sentiment'}
        )
    except Exception as e:
        print("Error loading news sentiment:", e)
        news_agg = pd.DataFrame(columns=['date', 'Ticker', 'news_finbert_sentiment'])

    # === Process Reddit Sentiment ===
    try:
        reddit = pd.read_csv(REDDIT_SENTIMENT)
        print("Loaded Reddit sentiment:", reddit.shape)
        reddit['date'] = pd.to_datetime(reddit['date']).dt.date if 'date' in reddit.columns else pd.to_datetime(reddit['created_utc'], unit='s').dt.date
        # Try to extract ticker from title if not present
        if 'Ticker' not in reddit.columns:
            reddit['Ticker'] = reddit['title'].apply(lambda x: extract_ticker_from_title(str(x), known_tickers))
        reddit['Ticker'] = reddit['Ticker'].str.upper()
        reddit['finbert_score'] = reddit['finbert_sentiment'].apply(sentiment_to_score)
        reddit_agg = reddit.groupby(['date', 'Ticker'], dropna=False)['finbert_score'].mean().reset_index().rename(
            columns={'finbert_score': 'reddit_finbert_sentiment'}
        )
    except Exception as e:
        print("Error loading Reddit sentiment:", e)
        reddit_agg = pd.DataFrame(columns=['date', 'Ticker', 'reddit_finbert_sentiment'])

    # === Merge Market + News + Reddit on Date and Ticker ===
    features = market.copy()
    if not news_agg.empty:
        features = features.merge(news_agg, left_on=['Date', 'Ticker'], right_on=['date', 'Ticker'], how='left')
    else:
        print("Warning: news_agg is empty. news_finbert_sentiment will be filled with NaN.")
    if not reddit_agg.empty:
        features = features.merge(reddit_agg, left_on=['Date', 'Ticker'], right_on=['date', 'Ticker'], how='left')
    else:
        print("Warning: reddit_agg is empty. reddit_finbert_sentiment will be filled with NaN.")
    # Drop unwanted duplicate date columns
    features.drop(columns=['date_x', 'date_y', 'date'], errors='ignore', inplace=True)
    # Ensure both sentiment columns exist
    if 'news_finbert_sentiment' not in features.columns:
        features['news_finbert_sentiment'] = np.nan
        print("Warning: news_finbert_sentiment column missing, filled with NaN.")
    if 'reddit_finbert_sentiment' not in features.columns:
        features['reddit_finbert_sentiment'] = np.nan
        print("Warning: reddit_finbert_sentiment column missing, filled with NaN.")
    # Fill missing sentiment with 0 (neutral)
    features['news_finbert_sentiment'] = features['news_finbert_sentiment'].fillna(0)
    features['reddit_finbert_sentiment'] = features['reddit_finbert_sentiment'].fillna(0)
    # Ensure output is not empty
    if features.shape[0] == 0:
        print("Error: Final features DataFrame is empty! Creating a single row of NaNs.")
        features = pd.DataFrame({col: [np.nan] for col in ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Ticker', 'Return', 'LogReturn', 'Volatility', 'Volume_MA', 'RelVolume', 'news_finbert_sentiment', 'reddit_finbert_sentiment']})
    # Save final output
    features.to_csv(PROCESSED_DATA, index=False)
    print(f"\n 197 All features (FinBERT + Technical) saved to: {PROCESSED_DATA}")
    print(f"Final shape: {features.shape}")
    print("Columns:", list(features.columns))

if __name__ == "__main__":
    main()

