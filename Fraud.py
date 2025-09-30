# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Fraud Prediction on Mobile Money Transaction",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-bottom: 1rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .fraud {
        background-color: #ffebee;
        border: 2px solid #f44336;
    }
    .genuine {
        background-color: #e8f5e8;
        border: 2px solid #4caf50;
    }
    .suspicious {
        background-color: #fff3e0;
        border: 2px solid #ff9800;
    }
    .risk-high {
        color: #f44336;
        font-weight: bold;
    }
    .risk-medium {
        color: #ff9800;
        font-weight: bold;
    }
    .risk-low {
        color: #4caf50;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Load model and resources
@st.cache_resource
def load_model():
    try:
        model = joblib.load('fraud_prediction.joblib')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Feature names (must match training)
feature_columns = [
    'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest',
    'balance_change_org', 'balance_change_dest', 'abs_balance_change_org',
    'type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER',
    'anomaly_zero_balance', 'anomaly_balance_mismatch'
]

def detect_advanced_anomalies(transaction_data, transaction_type):
    anomalies = []
    risk_factors = []
    
    amount = transaction_data['amount']
    oldbalance_org = transaction_data['oldbalanceOrg']
    newbalance_org = transaction_data['newbalanceOrig']
    oldbalance_dest = transaction_data['oldbalanceDest']
    newbalance_dest = transaction_data['newbalanceDest']
    
    balance_change_org = newbalance_org - oldbalance_org
    balance_change_dest = newbalance_dest - oldbalance_dest

    # --- existing checks omitted for brevity (kept as in previous version) ---

    tol = 1.0

    # A: Sender balance increases by more than amount
    if transaction_type in ["CASH_OUT", "TRANSFER", "PAYMENT"] and balance_change_org > amount + tol:
        anomalies.append("Sender balance increased more than the transaction amount")
        risk_factors.append(("Suspicious balance increase (> amount sent)", "HIGH"))

    # B: Sender balance increased ≈ amount
    if transaction_type in ["CASH_OUT", "TRANSFER", "PAYMENT"] and abs(balance_change_org - amount) <= tol:
        anomalies.append("Sender balance increased by approximately the transaction amount (expected decrease)")
        risk_factors.append(("Balance increased instead of decreasing", "HIGH"))

    # C: Insufficient funds but no debit
    if amount > oldbalance_org and transaction_type in ["CASH_OUT", "TRANSFER", "PAYMENT"]:
        if newbalance_org >= oldbalance_org:
            anomalies.append("Insufficient funds: attempted send but sender balance not debited")
            risk_factors.append(("Insufficient funds and no debit observed", "HIGH"))

    # D: Receiver gets nothing
    if transaction_type in ["TRANSFER", "CASH_OUT"] and amount > 0:
        if balance_change_dest <= 0:
            anomalies.append("Receiver got no funds despite sender sending money")
            risk_factors.append(("Receiver not credited", "HIGH"))

    # E: Sender unchanged but receiver credited
    if transaction_type in ["TRANSFER", "CASH_OUT", "PAYMENT"] and amount > 0:
        if abs(balance_change_org) <= tol and balance_change_dest >= amount:
            anomalies.append("Sender balance unchanged but receiver credited")
            risk_factors.append(("Suspicious non-debit with receiver credit", "HIGH"))

    # F: Sender loses more than sent
    if transaction_type in ["TRANSFER", "CASH_OUT", "PAYMENT"] and amount > 0:
        expected_newbalance = oldbalance_org - amount
        if newbalance_org < expected_newbalance - tol:
            hidden_loss = expected_newbalance - newbalance_org
            anomalies.append(f"Sender lost {hidden_loss:.2f} more than sent amount")
            risk_factors.append(("Excessive sender loss beyond transaction amount", "HIGH"))

    return anomalies, risk_factors

# --- rest of code unchanged ---
# In main(), inside anomaly mitigation actions section:
# Added recommendations for new conditions

# Run the app
if __name__ == "__main__":
    main()

# In mitigation section:
# if any("Sender balance unchanged but receiver credited" in anomaly for anomaly in anomalies):
#     st.write("• **Investigate accounting loophole** allowing credit without debit")
#     st.write("• **Audit transaction ledger** for unauthorized credit entries")

# if any("Sender lost" in anomaly for anomaly in anomalies):
#     st.write("• **Investigate missing funds** between debit and credit")
#     st.write("• **Check system logs** for rounding or duplicate debits")
#     st.write("• **Escalate to fraud team** if unexplained losses persist")
