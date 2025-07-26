import streamlit as st
import pandas as pd
import os
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scraper import fetch_stock_data
from engineer_features import compute_technical_features

# Page configuration
st.set_page_config(
    page_title="Anomaly Market Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    .anomaly-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .normal-card {
        background: linear-gradient(135deg, #2ed573 0%, #1e90ff 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .info-box {
        background: rgba(102, 126, 234, 0.1);
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .stSelectbox > div > div > div {
        background: white;
        border-radius: 10px;
    }
    
    .stTextInput > div > div > input {
        border-radius: 10px;
    }
    
    .stButton > button {
        border-radius: 10px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
    }
</style>
""", unsafe_allow_html=True)

PROCESSED_DATA = os.path.join('data', 'processed', 'anomaly_results.csv')
ALL_FEATURES = os.path.join('data', 'processed', 'all_features_bert.csv')
NEWS = os.path.join('data', 'raw', 'news_data.csv')
REDDIT = os.path.join('data', 'raw', 'reddit_data.csv')
MODEL_PATH = os.path.join('models', 'isolation_forest.pkl')
REQUIRED_FEATURES = ['Return', 'LogReturn', 'Volatility', 'RelVolume', 'news_finbert_sentiment', 'reddit_finbert_sentiment']

# Main header
st.markdown('<h1 class="main-header">🚀 Anomaly Market Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Advanced Machine Learning System for Market Anomaly Detection</p>', unsafe_allow_html=True)

# Sidebar for navigation and info
with st.sidebar:
    st.markdown("## 📊 Navigation")
    page = st.radio("Choose Analysis Type:", ["🔍 Real-time Analysis", "📈 Historical Data"])
    
    st.markdown("---")
    st.markdown("## ℹ️ How It Works")
    st.markdown("""
    **Isolation Forest Algorithm:**
    - Analyzes 6 key features
    - Identifies unusual patterns
    - Flags potential anomalies
    - Uses sentiment + technical data
    """)
    
    st.markdown("---")
    st.markdown("## 🎯 Features Analyzed")
    for feature in REQUIRED_FEATURES:
        st.markdown(f"• {feature}")

# Main content area
if page == "🔍 Real-time Analysis":
    st.markdown("## 🔍 Real-time Stock Analysis")
    st.markdown('<div class="info-box">Enter any stock ticker to analyze it in real-time using our anomaly detection model.</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        user_ticker = st.text_input("📈 Stock Ticker (e.g., INFY.NS, TCS.NS, RELIANCE.NS):", placeholder="Enter ticker symbol...")
    with col2:
        analyze_button = st.button("🚀 Analyze", use_container_width=True)
    
    user_df = None
    user_error = None
if user_ticker and analyze_button:
    with st.spinner(f"🔍 Analyzing {user_ticker}..."):
        try:
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("📥 Fetching market data...")
            progress_bar.progress(20)
            df = fetch_stock_data(user_ticker, period='1y', interval='1d')
            
            # Handle MultiIndex columns from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] if col[0] != 'Date' else 'Date' for col in df.columns]
            
            # Validate the DataFrame structure
            if df is None or df.empty:
                user_error = f"❌ No data found for ticker {user_ticker}."
            elif not isinstance(df, pd.DataFrame):
                user_error = f"❌ Invalid data format returned for ticker {user_ticker}."
            elif 'Close' not in df.columns or 'Volume' not in df.columns:
                user_error = f"❌ Missing required columns (Close, Volume) for ticker {user_ticker}."
            else:
                status_text.text("🔧 Processing data...")
                progress_bar.progress(40)
                
                # Ensure Date column exists and is properly formatted
                if 'Date' not in df.columns:
                    if df.index.name == 'Date':
                        df = df.reset_index()
                    else:
                        df['Date'] = df.index
                
                # Ensure Date is datetime
                df['Date'] = pd.to_datetime(df['Date'])
                
                # Add ticker column
                df['Ticker'] = user_ticker
                
                # Ensure numeric columns
                df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
                df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
                
                # Remove rows with invalid data
                df = df.dropna(subset=['Close', 'Volume'])
                
                if df.empty:
                    user_error = f"❌ No valid data after cleaning for ticker {user_ticker}."
                else:
                    status_text.text("🧠 Computing technical features...")
                    progress_bar.progress(60)
                    
                    df = compute_technical_features(df)
                    for col in ['news_finbert_sentiment', 'reddit_finbert_sentiment']:
                        if col not in df.columns:
                            df[col] = 0
                    df = df.dropna(subset=REQUIRED_FEATURES)
                    
                    if not df.empty:
                        # Check if all required features exist
                        missing_features = [col for col in REQUIRED_FEATURES if col not in df.columns]
                        if missing_features:
                            user_error = f"❌ Missing required features: {missing_features}"
                        else:
                            status_text.text("🤖 Running anomaly detection...")
                            progress_bar.progress(80)
                            
                            X = df[REQUIRED_FEATURES]
                            
                            # Check for any non-numeric data
                            for col in X.columns:
                                if not pd.api.types.is_numeric_dtype(X[col]):
                                    X[col] = pd.to_numeric(X[col], errors='coerce')
                            
                            # Remove any remaining NaN values
                            X = X.dropna()
                            if X.empty:
                                user_error = f"❌ No valid data after cleaning features for {user_ticker}."
                            else:
                                try:
                                    model = joblib.load(MODEL_PATH)
                                    
                                    # Ensure X is a numpy array or proper format for the model
                                    if hasattr(X, 'values'):
                                        X_values = X.values
                                    else:
                                        X_values = X
                                    
                                    predictions = model.predict(X_values)
                                    
                                    # Align predictions with the original DataFrame
                                    df_filtered = df.loc[X.index]
                                    df_filtered['anomaly'] = predictions
                                    user_df = df_filtered
                                    
                                    status_text.text("✅ Analysis complete!")
                                    progress_bar.progress(100)
                                    
                                except Exception as model_error:
                                    user_error = f"❌ Model prediction error: {model_error}"
                    else:
                        user_error = f"❌ Not enough data after feature engineering for {user_ticker}."
        except Exception as e:
            user_error = f"❌ Error processing {user_ticker}: {e}"
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()

if user_ticker and user_df is not None:
    # Success message
    st.success(f"✅ Analysis completed for {user_ticker}!")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        latest_price = user_df['Close'].iloc[-1]
        st.markdown(f'<div class="metric-card"><h3>💰 Latest Price</h3><h2>₹{latest_price:.2f}</h2></div>', unsafe_allow_html=True)
    
    with col2:
        total_anomalies = len(user_df[user_df['anomaly'] == -1])
        st.markdown(f'<div class="metric-card"><h3>🚨 Anomalies</h3><h2>{total_anomalies}</h2></div>', unsafe_allow_html=True)
    
    with col3:
        avg_volume = user_df['Volume'].mean()
        st.markdown(f'<div class="metric-card"><h3>📊 Avg Volume</h3><h2>{avg_volume:,.0f}</h2></div>', unsafe_allow_html=True)
    
    with col4:
        volatility = user_df['Volatility'].iloc[-1] if 'Volatility' in user_df.columns else 0
        st.markdown(f'<div class="metric-card"><h3>📈 Volatility</h3><h2>{volatility:.4f}</h2></div>', unsafe_allow_html=True)
    
    # Price and Volume Chart
    st.markdown("## 📈 Price & Volume Analysis")
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Stock Price', 'Trading Volume'),
        row_heights=[0.7, 0.3]
    )
    
    # Price chart
    fig.add_trace(
        go.Scatter(
            x=user_df['Date'],
            y=user_df['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='#667eea', width=2)
        ),
        row=1, col=1
    )
    
    # Volume chart
    fig.add_trace(
        go.Bar(
            x=user_df['Date'],
            y=user_df['Volume'],
            name='Volume',
            marker_color='#764ba2',
            opacity=0.7
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=600,
        showlegend=False,
        title_text=f"{user_ticker} - Price & Volume Analysis",
        title_x=0.5
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Anomaly Detection Results
    st.markdown("## 🚨 Anomaly Detection Results")
    
    # Create anomaly chart
    anomaly_fig = go.Figure()
    
    # Normal points
    normal_data = user_df[user_df['anomaly'] == 1]
    anomaly_fig.add_trace(go.Scatter(
        x=normal_data['Date'],
        y=normal_data['Close'],
        mode='markers',
        name='Normal',
        marker=dict(color='#2ed573', size=6, opacity=0.7)
    ))
    
    # Anomaly points
    anomaly_data = user_df[user_df['anomaly'] == -1]
    if not anomaly_data.empty:
        anomaly_fig.add_trace(go.Scatter(
            x=anomaly_data['Date'],
            y=anomaly_data['Close'],
            mode='markers',
            name='Anomaly',
            marker=dict(color='#ff6b6b', size=10, symbol='x')
        ))
    
    anomaly_fig.update_layout(
        title=f"Anomaly Detection for {user_ticker}",
        xaxis_title="Date",
        yaxis_title="Close Price",
        height=400
    )
    
    st.plotly_chart(anomaly_fig, use_container_width=True)
    
    # Anomaly details
    if not anomaly_data.empty:
        st.markdown("### 🚨 Flagged Anomalies")
        for idx, row in anomaly_data.iterrows():
            st.markdown(f"""
            <div class="anomaly-card">
                <strong>📅 Date:</strong> {row['Date'].strftime('%Y-%m-%d')} | 
                <strong>💰 Price:</strong> ₹{row['Close']:.2f} | 
                <strong>📊 Volume:</strong> {row['Volume']:,.0f}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="normal-card">
            <strong>✅ No anomalies detected!</strong> The stock appears to be trading normally.
        </div>
        """, unsafe_allow_html=True)
    
    # Feature Analysis
    st.markdown("## 🧠 Feature Analysis")
    
    if all(col in user_df.columns for col in ['Return', 'Volatility', 'RelVolume']):
        col1, col2 = st.columns(2)
        
        with col1:
            # Returns distribution
            fig_returns = px.histogram(
                user_df, 
                x='Return', 
                nbins=30,
                title='Returns Distribution',
                color_discrete_sequence=['#667eea']
            )
            st.plotly_chart(fig_returns, use_container_width=True)
        
        with col2:
            # Volatility over time
            fig_vol = px.line(
                user_df, 
                x='Date', 
                y='Volatility',
                title='Volatility Over Time',
                color_discrete_sequence=['#764ba2']
            )
            st.plotly_chart(fig_vol, use_container_width=True)

elif user_ticker and user_error:
    st.error(user_error)
elif page == "📈 Historical Data":
    st.markdown("## 📈 Historical Data Analysis")
    st.markdown('<div class="info-box">Analyze pre-processed historical data with sentiment analysis and anomaly detection.</div>', unsafe_allow_html=True)
    
    try:
        # Load data
        df = pd.read_csv(PROCESSED_DATA)
        features = pd.read_csv(ALL_FEATURES)
        news = pd.read_csv(NEWS)
        reddit = pd.read_csv(REDDIT)
        
        df['Date'] = pd.to_datetime(df['Date'])
        features['Date'] = pd.to_datetime(features['Date'])
        
        tickers = df['Ticker'].unique()
        
        # Stock selection
        col1, col2 = st.columns([2, 1])
        with col1:
            selected = st.selectbox("📊 Select Stock for Analysis", tickers)
        with col2:
            st.markdown(f"<div class='metric-card'><h4>Available Stocks</h4><h3>{len(tickers)}</h3></div>", unsafe_allow_html=True)
        
        if selected:
            stock_df = df[df['Ticker'] == selected]
            stock_features = features[features['Ticker'] == selected]
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                latest_price = stock_df['Close'].iloc[-1]
                st.markdown(f'<div class="metric-card"><h3>💰 Latest Price</h3><h2>₹{latest_price:.2f}</h2></div>', unsafe_allow_html=True)
            
            with col2:
                total_anomalies = len(stock_df[stock_df['anomaly'] == -1])
                st.markdown(f'<div class="metric-card"><h3>🚨 Anomalies</h3><h2>{total_anomalies}</h2></div>', unsafe_allow_html=True)
            
            with col3:
                avg_volume = stock_df['Volume'].mean()
                st.markdown(f'<div class="metric-card"><h3>📊 Avg Volume</h3><h2>{avg_volume:,.0f}</h2></div>', unsafe_allow_html=True)
            
            with col4:
                data_points = len(stock_df)
                st.markdown(f'<div class="metric-card"><h3>📅 Data Points</h3><h2>{data_points}</h2></div>', unsafe_allow_html=True)
            
            # Price and Volume Chart
            st.markdown("## 📈 Price & Volume Analysis")
            
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=('Stock Price', 'Trading Volume'),
                row_heights=[0.7, 0.3]
            )
            
            # Price chart
            fig.add_trace(
                go.Scatter(
                    x=stock_df['Date'],
                    y=stock_df['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='#667eea', width=2)
                ),
                row=1, col=1
            )
            
            # Volume chart
            fig.add_trace(
                go.Bar(
                    x=stock_df['Date'],
                    y=stock_df['Volume'],
                    name='Volume',
                    marker_color='#764ba2',
                    opacity=0.7
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                height=600,
                showlegend=False,
                title_text=f"{selected} - Historical Price & Volume",
                title_x=0.5
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Sentiment Analysis
            st.markdown("## 🎭 Sentiment Analysis")
            
            if not stock_features.empty and 'news_finbert_sentiment' in stock_features.columns:
                sentiment_fig = go.Figure()
                
                sentiment_fig.add_trace(go.Scatter(
                    x=stock_features['Date'],
                    y=stock_features['news_finbert_sentiment'],
                    mode='lines',
                    name='News Sentiment',
                    line=dict(color='#667eea', width=2)
                ))
                
                if 'reddit_finbert_sentiment' in stock_features.columns:
                    sentiment_fig.add_trace(go.Scatter(
                        x=stock_features['Date'],
                        y=stock_features['reddit_finbert_sentiment'],
                        mode='lines',
                        name='Reddit Sentiment',
                        line=dict(color='#764ba2', width=2)
                    ))
                
                sentiment_fig.update_layout(
                    title=f"Sentiment Analysis for {selected}",
                    xaxis_title="Date",
                    yaxis_title="Sentiment Score",
                    height=400
                )
                
                st.plotly_chart(sentiment_fig, use_container_width=True)
            
            # Anomaly Analysis
            st.markdown("## 🚨 Anomaly Analysis")
            
            # Create anomaly chart
            anomaly_fig = go.Figure()
            
            # Normal points
            normal_data = stock_df[stock_df['anomaly'] == 1]
            anomaly_fig.add_trace(go.Scatter(
                x=normal_data['Date'],
                y=normal_data['Close'],
                mode='markers',
                name='Normal',
                marker=dict(color='#2ed573', size=6, opacity=0.7)
            ))
            
            # Anomaly points
            anomaly_data = stock_df[stock_df['anomaly'] == -1]
            if not anomaly_data.empty:
                anomaly_fig.add_trace(go.Scatter(
                    x=anomaly_data['Date'],
                    y=anomaly_data['Close'],
                    mode='markers',
                    name='Anomaly',
                    marker=dict(color='#ff6b6b', size=10, symbol='x')
                ))
            
            anomaly_fig.update_layout(
                title=f"Anomaly Detection for {selected}",
                xaxis_title="Date",
                yaxis_title="Close Price",
                height=400
            )
            
            st.plotly_chart(anomaly_fig, use_container_width=True)
            
            # Anomaly details with context
            if not anomaly_data.empty:
                st.markdown("### 🚨 Flagged Anomalies with Context")
                for idx, row in anomaly_data.iterrows():
                    st.markdown(f"""
                    <div class="anomaly-card">
                        <strong>📅 Date:</strong> {row['Date'].strftime('%Y-%m-%d')} | 
                        <strong>💰 Price:</strong> ₹{row['Close']:.2f} | 
                        <strong>📊 Volume:</strong> {row['Volume']:,.0f}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # News context
                    news_on_date = news[news['title'].notnull() & (pd.to_datetime(news['date']).dt.date == row['Date'].date())]
                    if not news_on_date.empty:
                        st.markdown("**📰 Related News:**")
                        for _, nrow in news_on_date.iterrows():
                            st.markdown(f"• {nrow['title']}")
                    
                    # Reddit context
                    reddit_on_date = reddit[pd.to_datetime(reddit['created_utc'], unit='s').dt.date == row['Date'].date()]
                    if not reddit_on_date.empty:
                        st.markdown("**💬 Reddit Discussions:**")
                        for _, rrow in reddit_on_date.iterrows():
                            st.markdown(f"• {rrow['title']}")
                    
                    st.markdown("---")
            else:
                st.markdown("""
                <div class="normal-card">
                    <strong>✅ No anomalies detected!</strong> The stock appears to have traded normally during the analyzed period.
                </div>
                """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"❌ Error loading historical data: {e}")

# SHAP explainability (if available)
shap_path = os.path.join('data', 'processed', 'shap_values.csv')
if os.path.exists(shap_path):
    st.markdown("## 🧠 Model Explainability (SHAP)")
    st.markdown('<div class="info-box">SHAP (SHapley Additive exPlanations) values show the contribution of each feature to the model\'s predictions.</div>', unsafe_allow_html=True)
    
    try:
        shap_df = pd.read_csv(shap_path)
        
        # Feature importance summary
        feature_importance = shap_df[REQUIRED_FEATURES].abs().mean().sort_values(ascending=False)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### 📊 Feature Importance")
            for feature, importance in feature_importance.items():
                st.markdown(f"**{feature}:** {importance:.4f}")
        
        with col2:
            # SHAP values heatmap
            fig_shap = px.imshow(
                shap_df[REQUIRED_FEATURES].T,
                title="SHAP Values Heatmap",
                color_continuous_scale='RdBu',
                aspect='auto'
            )
            st.plotly_chart(fig_shap, use_container_width=True)
        
        # Detailed SHAP table
        with st.expander("📋 Detailed SHAP Values"):
            st.dataframe(shap_df)
            
    except Exception as e:
        st.error(f"❌ Error loading SHAP values: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p><strong>🚀 Anomaly Market Predictor</strong> - Advanced Machine Learning for Market Surveillance</p>
    <p>Built with ❤️ using Streamlit, Plotly, and Scikit-learn</p>
    <p>⚠️ This tool is for educational purposes only. Always consult financial advisors before making investment decisions.</p>
</div>
""", unsafe_allow_html=True)