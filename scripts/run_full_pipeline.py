import subprocess
import sys

steps = [
    ("Fetching market data", [sys.executable, "scripts/fetch_market_data.py"]),
    ("Fetching news data", [sys.executable, "scripts/fetch_news_data.py"]),
    ("Fetching Reddit data", [sys.executable, "scripts/fetch_social_data.py"]),
    ("Running FinBERT sentiment on news", [sys.executable, "scripts/sentiment_news_advanced.py"]),
    ("Running FinBERT sentiment on Reddit", [sys.executable, "scripts/sentiment_reddit_advanced.py"]),
    ("Engineering features", [sys.executable, "scripts/engineer_features.py"]),
    ("Training model and computing SHAP", [sys.executable, "scripts/train_model.py"]),
]

def main():
    user_ticker = None
    if len(sys.argv) > 1:
        user_ticker = sys.argv[1]
    for desc, cmd in steps:
        print(f"\n=== {desc} ===")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"Step failed: {desc}")
            sys.exit(result.returncode)
    if user_ticker:
        print(f"\n=== Predicting anomalies for user-specified ticker: {user_ticker} ===")
        result = subprocess.run([sys.executable, "scripts/predict_anomaly_for_stock.py", user_ticker])
        if result.returncode != 0:
            print(f"Prediction for {user_ticker} failed.")
            sys.exit(result.returncode)
    print("\nAll steps completed successfully!")
    print("To launch the dashboard, run:")
    print("  streamlit run scripts/dashboard.py")

if __name__ == "__main__":
    main() 