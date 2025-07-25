import os

# Stock tickers to monitor
TICKERS = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS']

# Data directories
RAW_DATA_DIR = os.path.join('data', 'raw')
PROCESSED_DATA_DIR = os.path.join('data', 'processed')
MODEL_DIR = 'models'

# News sources (for extension)
NEWS_SOURCES = [
    'https://www.moneycontrol.com/news/business/markets/',
    'https://economictimes.indiatimes.com/markets/stocks/news'
]

# Reddit subreddits
REDDIT_SUBS = ['IndiaInvesting', 'IndianStockMarket']

# YouTube search keywords
YOUTUBE_KEYWORDS = ['Indian stock market', 'NSE BSE stocks']

# Date range
START_DATE = '2022-01-01'
END_DATE = '2023-01-01'