# streamlit_app.py
import streamlit as st
from dashboard import predict_anomalies_for_ticker

st.title("📉 Stock Anomaly Detection")

ticker = st.text_input("Enter Stock Ticker (e.g. RELIANCE.NS):")

if ticker:
    try:
        result_df = predict_anomalies_for_ticker(ticker)
        st.success("✅ Anomaly prediction completed!")
        st.dataframe(result_df.tail(10))
    except ValueError as e:
        st.error(f"❌ Error: {e}")
