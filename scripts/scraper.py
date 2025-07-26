import yfinance as yf
import newspaper
import praw
import pandas as pd
from datetime import datetime, timedelta

# --- Stock Data Collector ---
def fetch_stock_data(symbol, period='1mo', interval='1d'):
    """Fetch historical stock data from Yahoo Finance."""
    try:
        data = yf.download(symbol, period=period, interval=interval, auto_adjust=True)
        
        # Check if data is empty or None
        if data is None or data.empty:
            print(f"No data returned for {symbol}")
            return pd.DataFrame()
        
        # Ensure we have a DataFrame
        if not isinstance(data, pd.DataFrame):
            print(f"Unexpected data type for {symbol}: {type(data)}")
            return pd.DataFrame()
        
        # Check for required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            print(f"Missing columns for {symbol}: {missing_columns}")
            return pd.DataFrame()
        
        # Reset index to make Date a column
        data.reset_index(inplace=True)
        
        # Ensure Date column exists
        if 'Date' not in data.columns:
            data['Date'] = data.index
        
        return data
        
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

# --- News Scraper ---
def scrape_news(query, max_articles=5):
    """Scrape news articles using newspaper3k for a given query."""
    # Example: Use Google News RSS or a list of URLs for demo
    urls = [f'https://news.google.com/rss/search?q={query}+stock']
    articles = []
    for url in urls:
        paper = newspaper.build(url, memoize_articles=False)
        for article in paper.articles[:max_articles]:
            try:
                article.download()
                article.parse()
                articles.append({
                    'title': article.title,
                    'text': article.text,
                    'published': article.publish_date
                })
            except Exception:
                continue
    return pd.DataFrame(articles)

# --- Reddit Scraper ---
def fetch_reddit_posts(subreddit, query, limit=10):
    """Fetch Reddit posts using PRAW for a given subreddit and query."""
    reddit = praw.Reddit(
        client_id='YOUR_CLIENT_ID',
        client_secret='YOUR_CLIENT_SECRET',
        user_agent='anomaly-detector'
    )
    posts = []
    for submission in reddit.subreddit(subreddit).search(query, limit=limit):
        posts.append({
            'title': submission.title,
            'score': submission.score,
            'created': datetime.fromtimestamp(submission.created_utc),
            'num_comments': submission.num_comments
        })
    return pd.DataFrame(posts)
