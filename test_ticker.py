import pandas as pd
import numpy as np
import joblib
import os
from scripts.scraper import fetch_stock_data
from scripts.engineer_features import compute_technical_features

def test_ticker(ticker="IEX.NS"):
    print(f"Testing ticker: {ticker}")
    
    # Step 1: Fetch data
    print("\n1. Fetching stock data...")
    df = fetch_stock_data(ticker, period='1y', interval='1d')
    print(f"   DataFrame shape: {df.shape}")
    print(f"   DataFrame columns: {list(df.columns)}")
    print(f"   DataFrame head:\n{df.head()}")

    # Flatten multi-index column names if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if col[0] != 'Date' else 'Date' for col in df.columns]
        print("   Flattened columns:", df.columns)
    
    # Step 2: Add ticker column
    print("\n2. Adding ticker column...")
    df['Ticker'] = ticker
    
    # Step 3: Ensure Date column
    if 'Date' not in df.columns:
        if df.index.name == 'Date':
            df = df.reset_index()
        else:
            df['Date'] = df.index
    
    df['Date'] = pd.to_datetime(df['Date'])
    print(f"   Date column type: {df['Date'].dtype}")
    
    # Step 4: Convert numeric columns
    print("\n3. Converting numeric columns...")
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
    print(f"   Close dtype: {df['Close'].dtype}")
    print(f"   Volume dtype: {df['Volume'].dtype}")
    
    # Step 5: Remove invalid data
    df = df.dropna(subset=['Close', 'Volume'])
    print(f"   Shape after removing NaN: {df.shape}")
    
    if df.empty:
        print("   ERROR: No valid data after cleaning")
        return
    
    # Step 6: Compute technical features
    print("\n4. Computing technical features...")
    df = compute_technical_features(df)
    print(f"   Shape after feature engineering: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    
    if df.empty:
        print("   ERROR: No data after feature engineering")
        return
    
    # Step 7: Add sentiment features
    print("\n5. Adding sentiment features...")
    for col in ['news_finbert_sentiment', 'reddit_finbert_sentiment']:
        if col not in df.columns:
            df[col] = 0
    print(f"   Final columns: {list(df.columns)}")
    
    # Step 8: Check required features
    REQUIRED_FEATURES = ['Return', 'LogReturn', 'Volatility', 'RelVolume', 'news_finbert_sentiment', 'reddit_finbert_sentiment']
    print(f"\n6. Checking required features: {REQUIRED_FEATURES}")
    
    missing_features = [col for col in REQUIRED_FEATURES if col not in df.columns]
    if missing_features:
        print(f"   ERROR: Missing features: {missing_features}")
        return
    
    # Step 9: Prepare features for model
    print("\n7. Preparing features for model...")
    df = df.dropna(subset=REQUIRED_FEATURES)
    print(f"   Shape after dropping NaN: {df.shape}")
    
    if df.empty:
        print("   ERROR: No data after dropping NaN")
        return
    
    X = df[REQUIRED_FEATURES]
    print(f"   X shape: {X.shape}")
    print(f"   X dtypes: {X.dtypes.to_dict()}")
    print(f"   X head:\n{X.head()}")
    
    # Step 10: Check for non-numeric data
    print("\n8. Checking for non-numeric data...")
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            print(f"   WARNING: Non-numeric column {col}: {X[col].dtype}")
            X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Step 11: Final cleaning
    X = X.dropna()
    print(f"   Final X shape: {X.shape}")
    
    if X.empty:
        print("   ERROR: No data after final cleaning")
        return
    
    # Step 12: Load model and predict
    print("\n9. Loading model and predicting...")
    MODEL_PATH = os.path.join('models', 'isolation_forest.pkl')
    
    try:
        model = joblib.load(MODEL_PATH)
        print(f"   Model type: {type(model)}")
        
        # Convert to numpy array
        X_values = X.values
        print(f"   X_values type: {type(X_values)}")
        print(f"   X_values shape: {X_values.shape}")
        
        predictions = model.predict(X_values)
        print(f"   Predictions shape: {predictions.shape}")
        print(f"   Predictions: {predictions}")
        
        print("\nSUCCESS: All steps completed successfully!")
        
    except Exception as e:
        print(f"   ERROR in model prediction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ticker("IEX.NS") 