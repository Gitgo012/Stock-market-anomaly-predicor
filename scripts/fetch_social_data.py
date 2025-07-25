import praw
import pandas as pd
import os
from dotenv import load_dotenv
from datetime import datetime, timezone

# Load environment variables from .env file
load_dotenv()

REDDIT_SUBS = ['IndiaInvesting', 'IndianStockMarket']
DATA_DIR = os.path.join('data', 'raw')
os.makedirs(DATA_DIR, exist_ok=True)

# List of tickers to search for
TICKERS = ["RELIANCE", "TCS", "HDFCBANK"]

# Reddit credentials
REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
REDDIT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT')

def fetch_reddit():
    reddit = praw.Reddit(client_id=REDDIT_CLIENT_ID,
                         client_secret=REDDIT_SECRET,
                         user_agent=REDDIT_USER_AGENT)

    posts = []
    for sub in REDDIT_SUBS:
        print(f"🔍 Fetching posts from r/{sub}")
        subreddit = reddit.subreddit(sub)
        for post in subreddit.new(limit=500):  # Adjust as needed
            title = post.title.lower()
            if any(ticker.lower() in title for ticker in TICKERS):
                post_date = datetime.fromtimestamp(post.created_utc, tz=timezone.utc).date()
                posts.append({
                    'subreddit': sub,
                    'title': post.title,
                    'created_utc': post.created_utc,
                    'date': post_date,
                    'score': post.score,
                    'url': post.url
                })

    df = pd.DataFrame(posts)
    if not df.empty:
        df.to_csv(os.path.join(DATA_DIR, 'reddit_data.csv'), index=False)
        print("✅ Filtered Reddit data saved to", os.path.join(DATA_DIR, 'reddit_data.csv'))
    else:
        print("⚠️ No matching posts found for given tickers.")

if __name__ == "__main__":
    fetch_reddit()


# Instructions:
# 1. Create a file named .env in your project root with the following content:
#    REDDIT_CLIENT_ID=your_client_id
#    REDDIT_SECRET=your_client_secret
#    REDDIT_USER_AGENT=anomaly-detector-script
# 2. Do NOT commit .env to version control (add to .gitignore) 