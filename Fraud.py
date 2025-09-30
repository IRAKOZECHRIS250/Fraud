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
    """Detect advanced fraud patterns and anomalies"""
    anomalies = []
    risk_factors = []
    
    amount = transaction_data['amount']
    oldbalance_org = transaction_data['oldbalanceOrg']
    newbalance_org = transaction_data['newbalanceOrig']
    oldbalance_dest = transaction_data['oldbalanceDest']
    newbalance_dest = transaction_data['newbalanceDest']
    
    # Calculate balance changes
    balance_change_org = newbalance_org - oldbalance_org
    balance_change_dest = newbalance_dest - oldbalance_dest
    
    # 1. Amount doesn't change in CASH_OUT transaction
    if transaction_type == "CASH_OUT" and amount == 0:
        anomalies.append("CASH_OUT transaction with zero amount")
        risk_factors.append(("Zero amount CASH_OUT", "HIGH"))
    
    # 2. Sender balance increases instead of decreases in debit transactions
    if transaction_type in ["CASH_OUT", "TRANSFER", "PAYMENT", "DEBIT"]:
        if balance_change_org > 0:
            anomalies.append(f"Sender balance increased in {transaction_type} transaction")
            risk_factors.append(("Sender balance increased in debit transaction", "HIGH"))
    
    # 3. Receiver balance decreases in CASH_IN transaction
    if transaction_type == "CASH_IN" and balance_change_dest < 0:
        anomalies.append("Receiver balance decreased in CASH_IN transaction")
        risk_factors.append(("Receiver balance decreased in CASH_IN", "HIGH"))
    
    # 4. No balance change in any transaction
    if balance_change_org == 0 and balance_change_dest == 0 and amount > 0:
        anomalies.append("No balance change despite transaction amount")
        risk_factors.append(("No balance change", "MEDIUM"))
    
    # 5. Large transaction with minimal balance change
    if amount > 10000 and abs(balance_change_org) < amount * 0.1:
        anomalies.append("Large transaction with minimal balance impact")
        risk_factors.append(("Large amount, small balance change", "MEDIUM"))
    
    # 6. Round number amounts (common in fraud)
    if amount % 1000 == 0 and amount > 1000:
        risk_factors.append(("Round number amount", "LOW"))
    
    # 7. Same account transaction
    if oldbalance_org == oldbalance_dest and newbalance_org == newbalance_dest:
        anomalies.append("Potential same account transaction")
        risk_factors.append(("Same account transaction", "MEDIUM"))
    
    # 8. Negative balances
    if newbalance_org < 0 or newbalance_dest < 0:
        anomalies.append("Negative balance detected")
        risk_factors.append(("Negative balance", "HIGH"))
    
    # 9. Transaction amount exceeds balance (insufficient funds)
    if amount > oldbalance_org and transaction_type in ["CASH_OUT", "TRANSFER", "PAYMENT"]:
        anomalies.append("Transaction amount exceeds sender balance (insufficient funds)")
        risk_factors.append(("Insufficient funds", "HIGH"))
    
    # 10. Unusual balance patterns
    if oldbalance_org == 0 and newbalance_org > 0 and transaction_type in ["CASH_OUT", "TRANSFER"]:
        anomalies.append("Zero to positive balance in debit transaction")
        risk_factors.append(("Zero to positive balance", "MEDIUM"))
    
    # 11. Money siphoning - sender sends more than receiver receives
    if transaction_type in ["TRANSFER", "CASH_OUT"]:
        expected_receiver_gain = amount
        actual_receiver_gain = balance_change_dest
        
        if actual_receiver_gain < expected_receiver_gain * 0.8 and actual_receiver_gain > 0:
            money_loss = expected_receiver_gain - actual_receiver_gain
            loss_percentage = (money_loss / expected_receiver_gain) * 100
            anomalies.append(f"Money siphoning detected: {loss_percentage:.1f}% of funds missing")
            risk_factors.append((f"Money siphoning ({loss_percentage:.1f}% loss)", "HIGH"))
        elif actual_receiver_gain <= 0 and amount > 0:
            anomalies.append("Receiver gets no money despite transaction")
            risk_factors.append(("Receiver gets zero funds", "HIGH"))

    # NEW CONDITIONS BASED ON USER REQUEST

    # Condition 3: Sender balance increases by more than the amount sent
    if transaction_type in ["CASH_OUT", "TRANSFER", "PAYMENT"] and balance_change_org > amount:
        anomalies.append("Sender balance increased more than the transaction amount")
        risk_factors.append(("Suspicious balance increase (> amount sent)", "HIGH"))

    # Condition 4: Sender balance increased exactly by the amount sent
    if transaction_type in ["CASH_OUT", "TRANSFER", "PAYMENT"] and balance_change_org == amount:
        anomalies.append("Sender balance increased equal to the transaction amount")
        risk_factors.append(("Balance increased instead of decreasing", "HIGH"))
    
    return anomalies, risk_factors

# =============================
# Keep rest of your original code here (preprocess_input, scale_features, predict_fraud, main(), etc.)
# =============================

# Run the app
if __name__ == "__main__":
    main()
