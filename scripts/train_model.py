import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
import os
import shap

PROCESSED_DATA = os.path.join('data', 'processed', 'all_features_bert.csv')
RESULTS_DATA = os.path.join('data', 'processed', 'anomaly_results.csv')
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'isolation_forest.pkl')
SHAP_PATH = os.path.join('data', 'processed', 'shap_values.csv')
os.makedirs(MODEL_DIR, exist_ok=True)

def main():
    df = pd.read_csv(PROCESSED_DATA)
    features = ['Return', 'LogReturn', 'Volatility', 'RelVolume', 'news_finbert_sentiment', 'reddit_finbert_sentiment']
    print("Columns in dataframe:", df.columns.tolist())
    print("Required features:", features)
    df = df.dropna(subset=features)
    X = df[features]
    model = IsolationForest(contamination=0.01, random_state=42)
    model.fit(X)
    df['anomaly'] = model.predict(X)
    df.to_csv(RESULTS_DATA, index=False)
    joblib.dump(model, MODEL_PATH)
    print(f"Model trained and results saved to {RESULTS_DATA}\nModel saved to {MODEL_PATH}")

    # SHAP explainability
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    shap_df = pd.DataFrame(shap_values.values, columns=features)
    shap_df['Date'] = df['Date'].values
    shap_df['Ticker'] = df['Ticker'].values
    shap_df.to_csv(SHAP_PATH, index=False)
    print(f"SHAP values saved to {SHAP_PATH}")

if __name__ == "__main__":
    main() 