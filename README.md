# 🚀 Anomaly Market Predictor

A comprehensive machine learning system for detecting pump and dump anomalies in stock markets using advanced sentiment analysis, technical indicators, and isolation forest algorithms.

## 📊 Overview

This project combines multiple data sources and machine learning techniques to identify potential market anomalies:

- **Market Data**: Historical stock prices and volume data from Yahoo Finance
- **News Sentiment**: Financial news sentiment analysis using FinBERT
- **Social Sentiment**: Reddit discussions sentiment analysis
- **Technical Indicators**: Price movements, volatility, and volume analysis
- **Anomaly Detection**: Isolation Forest algorithm for identifying unusual patterns

## 🏗️ Project Structure

```
Anomaly Market Predictor/
├── 📁 data/
│   ├── 📁 raw/                    # Raw data files
│   │   ├── market_data.csv        # Stock price data
│   │   ├── news_data.csv          # News articles
│   │   └── reddit_data.csv        # Reddit posts
│   └── 📁 processed/              # Processed data files
│       ├── all_features_bert.csv  # Combined features with sentiment
│       ├── anomaly_results.csv    # Anomaly detection results
│       ├── news_sentiment_bert.csv # News sentiment scores
│       ├── reddit_sentiment_bert.csv # Reddit sentiment scores
│       └── shap_values.csv        # Model explainability values
├── 📁 models/
│   └── isolation_forest.pkl       # Trained anomaly detection model
├── 📁 scripts/
│   ├── 🎯 dashboard.py            # Streamlit web dashboard
│   ├── 🔧 config.py               # Configuration settings
│   ├── 📈 fetch_market_data.py    # Stock data collection
│   ├── 📰 fetch_news_data.py      # News data collection
│   ├── 💬 fetch_social_data.py    # Reddit data collection
│   ├── 🧠 engineer_features.py    # Feature engineering
│   ├── 🎭 sentiment_news_advanced.py # News sentiment analysis
│   ├── 💭 sentiment_reddit_advanced.py # Reddit sentiment analysis
│   ├── 🤖 train_model.py          # Model training
│   ├── 🔮 predict_anomaly_for_stock.py # Individual stock prediction
│   ├── 🏃 run_full_pipeline.py    # Complete pipeline execution
│   └── 🕷️ scraper.py              # Data scraping utilities
├── 📋 requirements.txt            # Python dependencies
├── 🚫 .gitignore                  # Git ignore rules
└── 📖 README.md                   # This file
```

## 🚀 Features

### 📊 Data Collection
- **Real-time Market Data**: Fetches stock prices, volume, and technical indicators
- **News Sentiment**: Analyzes financial news using FinBERT model
- **Social Sentiment**: Processes Reddit discussions for market sentiment
- **Multi-source Integration**: Combines data from various sources

### 🧠 Machine Learning
- **Isolation Forest**: Unsupervised anomaly detection algorithm
- **Feature Engineering**: Technical indicators and sentiment features
- **SHAP Explainability**: Model interpretability and feature importance
- **Real-time Prediction**: Live anomaly detection for any stock ticker

### 🎨 User Interface
- **Streamlit Dashboard**: Interactive web interface
- **Real-time Charts**: Price movements and anomaly flags
- **Sentiment Visualization**: News and social sentiment trends
- **Anomaly Context**: Detailed information about flagged anomalies

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/anomaly-market-predictor.git
   cd anomaly-market-predictor
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download required models**
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   ```

## 🚀 Usage

### Quick Start

1. **Run the complete pipeline**
   ```bash
   python scripts/run_full_pipeline.py
   ```

2. **Launch the dashboard**
   ```bash
   streamlit run scripts/dashboard.py
   ```

3. **Access the web interface**
   - Open your browser and go to `http://localhost:8501`
   - Use the dropdown to select pre-analyzed stocks
   - Or enter any stock ticker for real-time analysis

### Individual Components

#### Data Collection
```bash
# Fetch market data
python scripts/fetch_market_data.py

# Fetch news data
python scripts/fetch_news_data.py

# Fetch Reddit data
python scripts/fetch_social_data.py
```

#### Sentiment Analysis
```bash
# Analyze news sentiment
python scripts/sentiment_news_advanced.py

# Analyze Reddit sentiment
python scripts/sentiment_reddit_advanced.py
```

#### Model Training
```bash
# Train the anomaly detection model
python scripts/train_model.py
```

#### Individual Stock Analysis
```bash
# Analyze a specific stock
python scripts/predict_anomaly_for_stock.py TCS.NS
```

## 📊 Configuration

Edit `scripts/config.py` to customize:

- **Stock Tickers**: Add or modify the stocks to monitor
- **Date Range**: Set the analysis period
- **News Sources**: Configure news websites to scrape
- **Reddit Subreddits**: Specify subreddits for social sentiment
- **Data Directories**: Modify file paths if needed

```python
# Example configuration
TICKERS = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS']
START_DATE = '2022-01-01'
END_DATE = '2023-01-01'
REDDIT_SUBS = ['IndiaInvesting', 'IndianStockMarket']
```

## 🔧 Technical Details

### Data Sources
- **Yahoo Finance**: Stock price and volume data
- **Financial News**: Market-related news articles
- **Reddit**: Social media discussions about stocks

### Machine Learning Pipeline
1. **Data Collection**: Fetch raw data from multiple sources
2. **Feature Engineering**: Calculate technical indicators and sentiment scores
3. **Data Preprocessing**: Clean and normalize data
4. **Model Training**: Train Isolation Forest on historical data
5. **Anomaly Detection**: Identify unusual patterns in new data
6. **Explainability**: Generate SHAP values for model interpretation

### Key Features
- **Return**: Daily price returns
- **LogReturn**: Logarithmic returns
- **Volatility**: Rolling standard deviation of returns
- **Relative Volume**: Volume compared to moving average
- **News Sentiment**: FinBERT sentiment scores from news
- **Social Sentiment**: FinBERT sentiment scores from Reddit

## 📈 Model Performance

The system uses an **Isolation Forest** algorithm with:
- **Contamination**: 1% (expected anomaly rate)
- **Features**: 6 engineered features
- **Explainability**: SHAP values for feature importance
- **Real-time**: Can analyze any stock ticker instantly

## 🎯 Use Cases

- **Market Surveillance**: Detect unusual trading patterns
- **Risk Management**: Identify potential market manipulation
- **Investment Research**: Analyze sentiment and technical factors
- **Regulatory Compliance**: Monitor for suspicious activities

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


## ⚠️ Disclaimer

This tool is for educational and research purposes only. It should not be used as the sole basis for investment decisions. Always conduct thorough research and consult with financial advisors before making investment decisions.

## 🐛 Troubleshooting

### Common Issues

1. **Data Download Errors**
   - Check internet connection
   - Verify stock ticker symbols
   - Ensure date range is valid

2. **Model Loading Issues**
   - Verify model file exists in `models/` directory
   - Check file permissions
   - Re-run training if needed

3. **Dashboard Issues**
   - Ensure Streamlit is installed: `pip install streamlit`
   - Check port availability (default: 8501)
   - Verify all dependencies are installed

### Getting Help

- Check the [Issues](https://github.com/yourusername/anomaly-market-predictor/issues) page
- Create a new issue with detailed error information
- Include system information and error logs

## 📊 Sample Output

The dashboard provides:
- **Price Charts**: Historical price movements
- **Anomaly Flags**: Red markers for detected anomalies
- **Sentiment Trends**: News and social sentiment over time
- **Feature Importance**: SHAP values showing key factors
- **Detailed Analysis**: Context for each flagged anomaly

## 🔮 Future Enhancements

- [ ] Real-time data streaming
- [ ] Additional ML models (LSTM, Transformer)
- [ ] More data sources (Twitter, Telegram)
- [ ] Advanced visualization options
- [ ] API endpoints for integration
- [ ] Mobile application
- [ ] Multi-language support

---
