import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
from urllib.parse import urljoin
from datetime import datetime

# ✅ Tickers or company names to search for
TICKERS = ["RELIANCE", "TCS", "HDFCBANK"]


NEWS_SOURCES = [
    'https://www.moneycontrol.com/news/business/markets/',
    'https://economictimes.indiatimes.com/markets/stocks/news'
]

DATA_DIR = os.path.join('data', 'raw')
os.makedirs(DATA_DIR, exist_ok=True)

def fetch_news():
    articles = []
    for url in NEWS_SOURCES:
        print(f"🔍 Scraping: {url}")
        try:
            resp = requests.get(url, timeout=10)
            soup = BeautifulSoup(resp.text, 'html.parser')

            for item in soup.find_all('a', href=True):
                title = item.get_text(strip=True)
                link = item['href']
                full_url = urljoin(url, link)

                if not title or len(title) < 20:
                    continue

                # ✅ Filter by ticker/company name
                if not any(ticker.lower() in title.lower() for ticker in TICKERS):
                    continue

                article_date = datetime.today().strftime('%Y-%m-%d')

                # Try to extract date if possible
                parent = item.find_parent()
                if parent:
                    date_tag = parent.find('span', class_='dateline') or parent.find('time')
                    if date_tag:
                        date_text = date_tag.get_text(strip=True)
                        parsed_date = pd.to_datetime(date_text, errors='coerce')
                        if pd.notnull(parsed_date):
                            article_date = parsed_date.strftime('%Y-%m-%d')

                articles.append({
                    'title': title,
                    'url': full_url,
                    'source': url,
                    'date': article_date
                })

        except Exception as e:
            print(f"❌ Error scraping {url}: {e}")

    df = pd.DataFrame(articles)
    output_path = os.path.join(DATA_DIR, 'news_data.csv')
    df.to_csv(output_path, index=False)
    print(f"✅ News data saved to: {output_path}")

if __name__ == "__main__":
    fetch_news()
