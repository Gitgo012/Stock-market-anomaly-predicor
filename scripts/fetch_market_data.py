import yfinance as yf
import pandas as pd
import os

TICKERS = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]


START_DATE = "2022-01-01"
END_DATE = "2025-07-25"

all_data = []

for ticker in TICKERS:
    print(f"\n📥 Fetching data for {ticker}...")
    df = yf.download(ticker, start=START_DATE, end=END_DATE, interval='1d', group_by='ticker', auto_adjust=False)

    if df.empty:
        print(f"⚠️ No data returned for {ticker}")
        continue

    print("📊 Columns returned:", df.columns)

    # If columns are MultiIndex (they usually are not with single ticker), handle gracefully
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df[ticker]
        except KeyError:
            print(f"❌ MultiIndex issue: Ticker {ticker} not found in df.columns")
            continue

    try:
        df = df.dropna(subset=['Close', 'Volume'])
        df["Ticker"] = ticker.replace(".NS", "")
        all_data.append(df)
    except KeyError as e:
        print(f"❌ Missing column: {e}")
        continue

# Create directory if it doesn’t exist
output_path = "data/raw"
os.makedirs(output_path, exist_ok=True)

# Combine and save to CSV
if all_data:
    final_df = pd.concat(all_data)
    final_df.reset_index(inplace=True)  
    final_df.to_csv(os.path.join(output_path, "market_data.csv"), index=False)
    print("\n Final combined data saved to data/raw/market_data.csv")
    print(final_df.head())
else:
    print("\n No data fetched for any ticker.")
