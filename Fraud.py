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
    
    tol = 1.0  # tolerance for float comparisons

    # Existing checks
    if transaction_type == "CASH_OUT" and amount == 0:
        anomalies.append("CASH_OUT transaction with zero amount")
        risk_factors.append(("Zero amount CASH_OUT", "HIGH"))
    
    if transaction_type in ["CASH_OUT", "TRANSFER", "PAYMENT", "DEBIT"]:
        if balance_change_org > 0:
            anomalies.append(f"Sender balance increased in {transaction_type} transaction")
            risk_factors.append(("Sender balance increased in debit transaction", "HIGH"))
    
    if transaction_type == "CASH_IN" and balance_change_dest < 0:
        anomalies.append("Receiver balance decreased in CASH_IN transaction")
        risk_factors.append(("Receiver balance decreased in CASH_IN", "HIGH"))
    
    if balance_change_org == 0 and balance_change_dest == 0 and amount > 0:
        anomalies.append("No balance change despite transaction amount")
        risk_factors.append(("No balance change", "MEDIUM"))
    
    if amount > 10000 and abs(balance_change_org) < amount * 0.1:
        anomalies.append("Large transaction with minimal balance impact")
        risk_factors.append(("Large amount, small balance change", "MEDIUM"))
    
    if amount % 1000 == 0 and amount > 1000:
        risk_factors.append(("Round number amount", "LOW"))
    
    if oldbalance_org == oldbalance_dest and newbalance_org == newbalance_dest:
        anomalies.append("Potential same account transaction")
        risk_factors.append(("Same account transaction", "MEDIUM"))
    
    if newbalance_org < 0 or newbalance_dest < 0:
        anomalies.append("Negative balance detected")
        risk_factors.append(("Negative balance", "HIGH"))
    
    if amount > oldbalance_org and transaction_type in ["CASH_OUT", "TRANSFER", "PAYMENT"]:
        anomalies.append("Transaction amount exceeds sender balance (insufficient funds)")
        risk_factors.append(("Insufficient funds", "HIGH"))
    
    if oldbalance_org == 0 and newbalance_org > 0 and transaction_type in ["CASH_OUT", "TRANSFER"]:
        anomalies.append("Zero to positive balance in debit transaction")
        risk_factors.append(("Zero to positive balance", "MEDIUM"))
    
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
    
    # --------------------
    # NEW FRAUD CONDITIONS
    # --------------------
    # E: Sender balance unchanged but receiver credited
    if amount > 0 and abs(balance_change_org) <= tol and balance_change_dest >= amount - tol:
        anomalies.append("Sender balance unchanged but receiver credited")
        risk_factors.append(("Unauthorized credit (sender not debited)", "HIGH"))

    # F: Sender balance decreases by more than the amount sent
    if amount > 0 and -balance_change_org > amount + tol and balance_change_dest >= amount - tol:
        anomalies.append("Sender lost more than the amount sent (hidden loss)")
        risk_factors.append(("Excess debit beyond sent amount", "HIGH"))

    return anomalies, risk_factors


def calculate_risk_score(risk_factors):
    risk_weights = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
    total_score = 0
    max_score = 0
    for factor, level in risk_factors:
        total_score += risk_weights[level]
        max_score += risk_weights["HIGH"]
    if max_score == 0:
        return 0
    return (total_score / max_score) * 100


def preprocess_input(transaction_data, transaction_type):
    features = {}
    features['amount'] = transaction_data['amount']
    features['oldbalanceOrg'] = transaction_data['oldbalanceOrg']
    features['newbalanceOrig'] = transaction_data['newbalanceOrig']
    features['oldbalanceDest'] = transaction_data['oldbalanceDest']
    features['newbalanceDest'] = transaction_data['newbalanceDest']
    features['balance_change_org'] = features['newbalanceOrig'] - features['oldbalanceOrg']
    features['balance_change_dest'] = features['newbalanceDest'] - features['oldbalanceDest']
    features['abs_balance_change_org'] = abs(features['balance_change_org'])
    for ttype in ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']:
        features[f'type_{ttype}'] = 1 if transaction_type == ttype else 0
    features['anomaly_zero_balance'] = 1 if (features['oldbalanceOrg'] == 0 and features['amount'] > 0) else 0
    expected_newbalance = features['oldbalanceOrg'] - features['amount']
    features['anomaly_balance_mismatch'] = 1 if abs(features['newbalanceOrig'] - expected_newbalance) > 1 else 0
    return pd.DataFrame([features])[feature_columns]


def scale_features(input_df):
    scaled_df = input_df.copy()
    for col in ['amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest']:
        scaled_df[col] = np.log1p(np.maximum(scaled_df[col], 0))
    for col in scaled_df.columns:
        if scaled_df[col].std() > 0:
            scaled_df[col] = (scaled_df[col]-scaled_df[col].mean())/scaled_df[col].std()
    return scaled_df


def predict_fraud(transaction_data, transaction_type, model):
    try:
        input_df = preprocess_input(transaction_data, transaction_type)
        scaled_input = scale_features(input_df)
        prediction = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0][1]
        anomalies, risk_factors = detect_advanced_anomalies(transaction_data, transaction_type)
        risk_score = calculate_risk_score(risk_factors)
        return prediction, probability, input_df, anomalies, risk_factors, risk_score
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None, None, None, [], [], 0


def main():
    st.markdown('<div class="main-header">💰 Fraud Prediction on Mobile Money Transaction</div>', unsafe_allow_html=True)
    st.markdown("This AI-powered system detects fraudulent transactions in real-time.")
    model = load_model()
    if model is None:
        st.error("Failed to load model")
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sub-header">📊 Model Information</div>', unsafe_allow_html=True)
        st.info("**Model**: Logistic Regression\n**Dataset**: PaySim\n**Accuracy**: ~99.8%")
        st.markdown('<div class="sub-header">🔧 Settings</div>', unsafe_allow_html=True)
        fraud_threshold = st.slider("Fraud Threshold",0.1,0.9,0.3,0.1)
        enable_advanced_rules = st.checkbox("Enable Advanced Rules", value=True)
    
    col1, col2 = st.columns(2)
    with col1:
        transaction_type = st.selectbox("Transaction Type",["CASH_IN","CASH_OUT","DEBIT","PAYMENT","TRANSFER"])
        amount = st.number_input("Transaction Amount",0.0,1000000.0,1000.0,100.0)
        oldbalance_org = st.number_input("Old Balance - Sender",0.0,1000000.0,5000.0,500.0)
        newbalance_org = st.number_input("New Balance - Sender",0.0,1000000.0,4000.0,500.0)
    with col2:
        oldbalance_dest = st.number_input("Old Balance - Receiver",0.0,1000000.0,3000.0,500.0)
        newbalance_dest = st.number_input("New Balance - Receiver",0.0,1000000.0,4000.0,500.0)

        st.markdown("**Real-time Risk Assessment**")
        balance_change_org = newbalance_org - oldbalance_org
        balance_change_dest = newbalance_dest - oldbalance_dest
        tol=1.0
        risk_items=[]
        if oldbalance_org==0 and amount>0: risk_items.append(("Zero balance","HIGH","🔴"))
        expected_balance=oldbalance_org-amount
        if abs(newbalance_org-expected_balance)>1: risk_items.append(("Balance mismatch","HIGH","🔴"))
        if transaction_type in ["CASH_OUT","TRANSFER","PAYMENT"] and balance_change_org>0:
            risk_items.append(("Sender balance increased","HIGH","🔴"))
        if balance_change_org==0 and balance_change_dest==0 and amount>0:
            risk_items.append(("No balance change","MEDIUM","🟡"))
        if transaction_type in ["TRANSFER","CASH_OUT"]:
            expected_receiver_gain=amount
            actual_receiver_gain=balance_change_dest
            if actual_receiver_gain<expected_receiver_gain*0.8 and actual_receiver_gain>0:
                loss_percentage=(expected_receiver_gain-actual_receiver_gain)/expected_receiver_gain*100
                risk_items.append((f"Money siphoning ({loss_percentage:.1f}% loss)","HIGH","🔴"))
            elif actual_receiver_gain<=0 and amount>0:
                risk_items.append(("Receiver gets no funds","HIGH","🔴"))
        # NEW checks real-time
        if amount>0 and abs(balance_change_org)<=tol and balance_change_dest>=amount-tol:
            risk_items.append(("Sender unchanged but receiver credited","HIGH","🔴"))
        if amount>0 and -balance_change_org>amount+tol and balance_change_dest>=amount-tol:
            risk_items.append(("Sender lost more than sent","HIGH","🔴"))
        
        if risk_items:
            for risk,level,icon in risk_items: st.write(f"{icon} {risk} - {level}")
        else: st.write("✅ No immediate risks detected")
    
    if st.button("🚀 Check for Fraud"):
        with st.spinner("Analyzing..."):
            transaction_data={'amount':amount,'oldbalanceOrg':oldbalance_org,'newbalanceOrig':newbalance_org,
                              'oldbalanceDest':oldbalance_dest,'newbalanceDest':newbalance_dest}
            prediction,probability,features_df,anomalies,risk_factors,risk_score= predict_fraud(transaction_data,transaction_type,model)
            if prediction is not None:
                st.markdown("---")
                overall_fraud = (probability>=fraud_threshold) or (enable_advanced_rules and len(anomalies)>0 and risk_score>50)
                if overall_fraud:
                    st.error(f"🚨 FRAUD DETECTED! (ML={probability:.3f}, Risk={risk_score:.1f}%)")
                else:
                    st.success(f"✅ Genuine Transaction (ML={probability:.3f}, Risk={risk_score:.1f}%)")
                if anomalies:
                    with st.expander("🛡️ Specific Risk Mitigation Actions"):
                        if any("unchanged but receiver credited" in a for a in anomalies):
                            st.write("• Audit system for unauthorized credit without debit")
                        if any("lost more than the amount sent" in a for a in anomalies):
                            st.write("• Investigate missing funds, check logs, escalate to fraud team")

if __name__=="__main__":
    main()
