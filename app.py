import streamlit as st
import pandas as pd
import joblib

# Page settings
st.set_page_config(page_title="Gold Price Predictor", page_icon="💰", layout="centered")

st.title("💰 Gold Price Prediction System")
st.markdown("### Using Random Forest Regressor")

# Load the model
@st.cache_resource
def load_model():
    return joblib.load('gold_price_model.pkl')

model = load_model()

st.sidebar.header("📊 Enter Current Market Values")

spx = st.sidebar.number_input("SPX (S&P 500 Index)", value=2500.0, step=0.1, format="%.2f")
uso = st.sidebar.number_input("USO (Crude Oil Price)", value=50.0, step=0.1, format="%.2f")
slv = st.sidebar.number_input("SLV (Silver Price)", value=20.0, step=0.1, format="%.2f")
eur_usd = st.sidebar.number_input("EUR/USD Exchange Rate", 
                                  value=1.10, 
                                  step=0.001, 
                                  format="%.4f", 
                                  min_value=0.5, 
                                  max_value=2.0)

if st.sidebar.button("🔮 Predict Gold Price"):
    input_data = pd.DataFrame({
        'SPX': [spx],
        'USO': [uso],
        'SLV': [slv],
        'EUR/USD': [eur_usd]
    })
    
    prediction = model.predict(input_data)[0]
    
    st.success(f"**Predicted Gold Price (GLD): ${prediction:.2f}**")
    
    st.info(f"""
    **Input Values:**
    - SPX: {spx}
    - USO: {uso}
    - SLV: {slv}
    - EUR/USD: {eur_usd:.4f}
    """)
st.markdown("---")
st.caption("Project by [Amna Ejaz] | Hamdard University")