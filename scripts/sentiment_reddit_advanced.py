import pandas as pd
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

RAW_REDDIT = os.path.join('data', 'raw', 'reddit_data.csv')
PROCESSED_REDDIT = os.path.join('data', 'processed', 'reddit_sentiment_bert.csv')
os.makedirs(os.path.dirname(PROCESSED_REDDIT), exist_ok=True)

def main():
    df = pd.read_csv(RAW_REDDIT)
    tokenizer = AutoTokenizer.from_pretrained('yiyanghkust/finbert-tone')
    model = AutoModelForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', use_safetensors=True)
    nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
    df['finbert_sentiment'] = df['title'].astype(str).apply(lambda x: nlp(x)[0]['label'])
    df.to_csv(PROCESSED_REDDIT, index=False)
    print(f"FinBERT reddit sentiment saved to {PROCESSED_REDDIT}")

if __name__ == "__main__":
    main() 