import pandas as pd
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

RAW_NEWS = os.path.join('data', 'raw', 'news_data.csv')
PROCESSED_NEWS = os.path.join('data', 'processed', 'news_sentiment_bert.csv')
os.makedirs(os.path.dirname(PROCESSED_NEWS), exist_ok=True)

def main():
    df = pd.read_csv(RAW_NEWS)
    tokenizer = AutoTokenizer.from_pretrained('yiyanghkust/finbert-tone')
    model = AutoModelForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', use_safetensors=True)
    nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
    df['finbert_sentiment'] = df['title'].astype(str).apply(lambda x: nlp(x)[0]['label'])
    df.to_csv(PROCESSED_NEWS, index=False)
    print(f"FinBERT news sentiment saved to {PROCESSED_NEWS}")

if __name__ == "__main__":
    main() 