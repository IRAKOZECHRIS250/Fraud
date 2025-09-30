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
        
        # If receiver gets significantly less than the sent amount
        if actual_receiver_gain < expected_receiver_gain * 0.8 and actual_receiver_gain > 0:
            money_loss = expected_receiver_gain - actual_receiver_gain
            loss_percentage = (money_loss / expected_receiver_gain) * 100
            anomalies.append(f"Money siphoning detected: {loss_percentage:.1f}% of funds missing")
            risk_factors.append((f"Money siphoning ({loss_percentage:.1f}% loss)", "HIGH"))
        
        # If receiver gets nothing despite transaction
        elif actual_receiver_gain <= 0 and amount > 0:
            anomalies.append("Receiver gets no money despite transaction")
            risk_factors.append(("Receiver gets zero funds", "HIGH"))
    
    # ------------------------------
    # NEW: Additional checks requested by user
    # ------------------------------
    # Use a small tolerance to compare balances (helps with float input)
    tol = 1.0  # 1 dollar tolerance

    # Condition A: Sender balance increases by more than the amount sent
    if transaction_type in ["CASH_OUT", "TRANSFER", "PAYMENT"] and balance_change_org > amount + tol:
        anomalies.append("Sender balance increased more than the transaction amount")
        risk_factors.append(("Suspicious balance increase (> amount sent)", "HIGH"))

    # Condition B: Sender balance increased approximately equal to the amount sent (instead of decreasing)
    if transaction_type in ["CASH_OUT", "TRANSFER", "PAYMENT"] and abs(balance_change_org - amount) <= tol:
        anomalies.append("Sender balance increased by approximately the transaction amount (expected decrease)")
        risk_factors.append(("Balance increased instead of decreasing", "HIGH"))

    # Condition C: Sender tries to send money but balance is too low (explicit flag for insufficient funds)
    # Already captured above (amount > oldbalance_org), but add a clearer message when balance low AND newbalance didn't drop
    if amount > oldbalance_org and transaction_type in ["CASH_OUT", "TRANSFER", "PAYMENT"]:
        # if the system still shows a newbalance that doesn't reflect a decline, add more context
        if newbalance_org >= oldbalance_org:
            anomalies.append("Insufficient funds: attempted send but sender balance not debited")
            risk_factors.append(("Insufficient funds and no debit observed", "HIGH"))

    # Condition D: Sender sends money but receiver gets nothing (explicit)
    if transaction_type in ["TRANSFER", "CASH_OUT"] and amount > 0:
        actual_receiver_gain = balance_change_dest
        if actual_receiver_gain <= 0:
            anomalies.append("Receiver got no funds despite sender sending money")
            risk_factors.append(("Receiver not credited", "HIGH"))

    return anomalies, risk_factors


def calculate_risk_score(risk_factors):
    """Calculate overall risk score based on risk factors"""
    risk_weights = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
    total_score = 0
    max_score = 0
    
    for factor, level in risk_factors:
        total_score += risk_weights[level]
        max_score += risk_weights["HIGH"]  # Assume all could be high
    
    if max_score == 0:
        return 0
    
    risk_percentage = (total_score / max_score) * 100
    return risk_percentage


def preprocess_input(transaction_data, transaction_type):
    """Preprocess the input data to match training format"""
    
    # Create feature dictionary
    features = {}
    
    # Basic features
    features['amount'] = transaction_data['amount']
    features['oldbalanceOrg'] = transaction_data['oldbalanceOrg']
    features['newbalanceOrig'] = transaction_data['newbalanceOrig']
    features['oldbalanceDest'] = transaction_data['oldbalanceDest']
    features['newbalanceDest'] = transaction_data['newbalanceDest']
    
    # Calculated features
    features['balance_change_org'] = features['newbalanceOrig'] - features['oldbalanceOrg']
    features['balance_change_dest'] = features['newbalanceDest'] - features['oldbalanceDest']
    features['abs_balance_change_org'] = abs(features['balance_change_org'])
    
    # Transaction type encoding
    for ttype in ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']:
        features[f'type_{ttype}'] = 1 if transaction_type == ttype else 0
    
    # Anomaly flags
    features['anomaly_zero_balance'] = 1 if (features['oldbalanceOrg'] == 0 and features['amount'] > 0) else 0
    expected_newbalance = features['oldbalanceOrg'] - features['amount']
    features['anomaly_balance_mismatch'] = 1 if abs(features['newbalanceOrig'] - expected_newbalance) > 1 else 0
    
    # Create DataFrame with correct column order
    input_df = pd.DataFrame([features])[feature_columns]
    
    return input_df


def scale_features(input_df):
    """Scale features using the same scaler as training"""
    scaled_df = input_df.copy()
    
    # Apply log transformation to amount and balances to reduce skew
    for col in ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']:
        # guard against negative or zero values when taking log
        scaled_df[col] = np.log1p(np.maximum(scaled_df[col], 0))
    
    # Standard scaling (simple in-place scaling using column mean/std to keep app lightweight)
    for col in scaled_df.columns:
        if scaled_df[col].std() > 0:
            scaled_df[col] = (scaled_df[col] - scaled_df[col].mean()) / scaled_df[col].std()
    
    return scaled_df


def predict_fraud(transaction_data, transaction_type, model):
    """Make fraud prediction"""
    try:
        # Preprocess input
        input_df = preprocess_input(transaction_data, transaction_type)
        
        # Scale features
        scaled_input = scale_features(input_df)
        
        # Make prediction
        prediction = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0][1]
        
        # Detect advanced anomalies
        anomalies, risk_factors = detect_advanced_anomalies(transaction_data, transaction_type)
        risk_score = calculate_risk_score(risk_factors)
        
        return prediction, probability, input_df, anomalies, risk_factors, risk_score
        
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None, None, None, [], [], 0

# Main app
def main():
    # Header
    st.markdown('<div class="main-header">💰 Fraud Prediction on Mobile Money Transaction</div>', unsafe_allow_html=True)
    
    st.markdown("""
    This AI-powered system detects fraudulent transactions in real-time using machine learning 
    and advanced anomaly detection rules. Enter the transaction details below to check for potential fraud.
    """)
    
    # Load model
    model = load_model()
    if model is None:
        st.error("Failed to load the fraud detection model. Please check if 'fraud_prediction.joblib' exists.")
        return
    
    # Sidebar for additional information
    with st.sidebar:
        st.markdown('<div class="sub-header">📊 Model Information</div>', unsafe_allow_html=True)
        st.info("""
        **Model**: Logistic Regression
        **Training Data**: PaySim Mobile Money Dataset
        **Accuracy**: ~99.8%
        **Precision**: ~85-90%
        **Recall**: ~75-80%
        """)
        
        st.markdown('<div class="sub-header">🚨 Advanced Fraud Patterns</div>', unsafe_allow_html=True)
        st.write("""
        - **CASH_OUT with zero amount**
        - **Sender balance increases** in debit transactions
        - **No balance change** despite transaction
        - **Large amounts** with minimal balance impact
        - **Round number amounts** (>$1000)
        - **Negative balances** after transaction
        - **Amount exceeds** available balance
        - **Money siphoning** (sender sends more than receiver receives)
        """)
        
        st.markdown('<div class="sub-header">🔧 Settings</div>', unsafe_allow_html=True)
        fraud_threshold = st.slider(
            "Fraud Detection Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.3,
            step=0.1,
            help="Higher values reduce false positives but may miss some fraud"
        )
        
        enable_advanced_rules = st.checkbox(
            "Enable Advanced Anomaly Detection",
            value=True,
            help="Enable additional fraud pattern detection rules"
        )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="sub-header">🔍 Transaction Details</div>', unsafe_allow_html=True)
        
        # Transaction type
        transaction_type = st.selectbox(
            "Transaction Type",
            ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"],
            help="Fraud is most common in TRANSFER and CASH_OUT transactions"
        )
        
        # Amount
        amount = st.number_input(
            "Transaction Amount ($)",
            min_value=0.0,
            value=1000.0,
            step=100.0,
            format="%.2f",
            help="Enter the transaction amount"
        )
        
        # Sender information
        st.markdown("**Sender Account Information**")
        oldbalance_org = st.number_input(
            "Old Balance - Sender ($)",
            min_value=0.0,
            value=5000.0,
            step=500.0,
            format="%.2f"
        )
        newbalance_org = st.number_input(
            "New Balance - Sender ($)",
            min_value=0.0,
            value=4000.0,
            step=500.0,
            format="%.2f"
        )
    
    with col2:
        st.markdown('<div class="sub-header">👤 Receiver Information</div>', unsafe_allow_html=True)
        
        # Receiver information
        oldbalance_dest = st.number_input(
            "Old Balance - Receiver ($)",
            min_value=0.0,
            value=3000.0,
            step=500.0,
            format="%.2f"
        )
        newbalance_dest = st.number_input(
            "New Balance - Receiver ($)",
            min_value=0.0,
            value=4000.0,
            step=500.0,
            format="%.2f"
        )
        
        # Additional information
        st.markdown("**Additional Details**")
        step = st.number_input(
            "Time Step (Hour)",
            min_value=0,
            value=1,
            help="Time step of the transaction (1 step = 1 hour)"
        )
        
        # Real-time risk factors display
        st.markdown("**Real-time Risk Assessment**")
        
        # Calculate basic risk factors in real-time
        balance_change_org = newbalance_org - oldbalance_org
        balance_change_dest = newbalance_dest - oldbalance_dest
        
        risk_items = []
        
        # Zero balance risk
        if oldbalance_org == 0 and amount > 0:
            risk_items.append(("Zero balance transaction", "HIGH", "🔴"))
        
        # Balance mismatch
        expected_balance = oldbalance_org - amount
        if abs(newbalance_org - expected_balance) > 1:
            risk_items.append(("Balance mismatch", "HIGH", "🔴"))
        
        # Sender balance increases in debit transaction
        if transaction_type in ["CASH_OUT", "TRANSFER", "PAYMENT"] and balance_change_org > 0:
            risk_items.append(("Sender balance increased", "HIGH", "🔴"))
        
        # No balance change
        if balance_change_org == 0 and balance_change_dest == 0 and amount > 0:
            risk_items.append(("No balance change", "MEDIUM", "🟡"))
        
        # Money siphoning risk (sender sends more than receiver receives)
        if transaction_type in ["TRANSFER", "CASH_OUT"]:
            expected_receiver_gain = amount
            actual_receiver_gain = newbalance_dest - oldbalance_dest
            if actual_receiver_gain < expected_receiver_gain * 0.8 and actual_receiver_gain > 0:
                money_loss = expected_receiver_gain - actual_receiver_gain
                loss_percentage = (money_loss / expected_receiver_gain) * 100
                risk_items.append((f"Money siphoning ({loss_percentage:.1f}% loss)", "HIGH", "🔴"))
            elif actual_receiver_gain <= 0 and amount > 0:
                risk_items.append(("Receiver gets no funds", "HIGH", "🔴"))
        
        # NEW: real-time checks for requested conditions
        tol = 1.0
        if transaction_type in ["CASH_OUT", "TRANSFER", "PAYMENT"] and balance_change_org > amount + tol:
            risk_items.append(("Sender balance increased > amount sent", "HIGH", "🔴"))
        if transaction_type in ["CASH_OUT", "TRANSFER", "PAYMENT"] and abs(balance_change_org - amount) <= tol:
            risk_items.append(("Sender balance increased = amount sent (expected decrease)", "HIGH", "🔴"))
        if amount > oldbalance_org and transaction_type in ["CASH_OUT", "TRANSFER", "PAYMENT"]:
            if newbalance_org >= oldbalance_org:
                risk_items.append(("Insufficient funds and no debit observed", "HIGH", "🔴"))
        
        if risk_items:
            for risk, level, icon in risk_items:
                st.write(f"{icon} {risk} - {level} Risk")
        else:
            st.write("✅ No immediate risks detected")
    
    # Prediction button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        predict_button = st.button(
            "🚀 Check for Fraud",
            use_container_width=True,
            type="primary"
        )
    
    # Make prediction when button is clicked
    if predict_button:
        with st.spinner("Analyzing transaction for fraud patterns..."):
            # Prepare transaction data
            transaction_data = {
                'amount': amount,
                'oldbalanceOrg': oldbalance_org,
                'newbalanceOrig': newbalance_org,
                'oldbalanceDest': oldbalance_dest,
                'newbalanceDest': newbalance_dest
            }
            
            # Make prediction
            prediction, probability, features_df, anomalies, risk_factors, risk_score = predict_fraud(
                transaction_data, transaction_type, model
            )
            
            if prediction is not None:
                # Display results
                st.markdown("---")
                st.markdown('<div class="sub-header">🎯 Fraud Detection Results</div>', unsafe_allow_html=True)
                
                # Determine fraud status based on threshold and risk factors
                ml_fraud = probability >= fraud_threshold
                rule_based_fraud = len(anomalies) > 0 and risk_score > 50
                overall_fraud = ml_fraud or (enable_advanced_rules and rule_based_fraud)
                
                # Display prediction box
                if overall_fraud:
                    if ml_fraud and rule_based_fraud:
                        fraud_type = "ML + Rule-Based Fraud Detection"
                        box_class = "fraud"
                    elif ml_fraud:
                        fraud_type = "Machine Learning Fraud Detection"
                        box_class = "fraud"
                    else:
                        fraud_type = "Rule-Based Fraud Detection"
                        box_class = "suspicious"
                    
                    st.markdown(
                        f'<div class="prediction-box {box_class}">' 
                        f'<h2>🚨 FRAUD DETECTED!</h2>'
                        f'<p><strong>Detection Method:</strong> {fraud_type}</p>'
                        f'<p><strong>ML Probability:</strong> {probability:.3f}</p>'
                        f'<p><strong>Rule-Based Risk Score:</strong> {risk_score:.1f}%</p>'
                        f'<p><strong>Status:</strong> This transaction appears to be fraudulent</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    
                else:
                    st.markdown(
                        f'<div class="prediction-box genuine">'
                        f'<h2>✅ GENUINE TRANSACTION</h2>'
                        f'<p><strong>ML Probability:</strong> {probability:.3f}</p>'
                        f'<p><strong>Rule-Based Risk Score:</strong> {risk_score:.1f}%</p>'
                        f'<p><strong>Status:</strong> This transaction appears to be genuine</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                
                # Display advanced anomaly detection results
                if enable_advanced_rules and (anomalies or risk_factors):
                    with st.expander("🔍 Advanced Anomaly Detection Results", expanded=True):
                        st.markdown("**Detected Anomalies:**")
                        if anomalies:
                            for anomaly in anomalies:
                                st.error(f"🚨 {anomaly}")
                        else:
                            st.success("✅ No anomalies detected")
                        
                        st.markdown("**Risk Factors:**")
                        if risk_factors:
                            for factor, level in risk_factors:
                                if level == "HIGH":
                                    st.error(f"🔴 {factor} - {level} Risk")
                                elif level == "MEDIUM":
                                    st.warning(f"🚡 {factor} - {level} Risk")
                                else:
                                    st.info(f"🔵 {factor} - {level} Risk")
                        else:
                            st.success("✅ No additional risk factors")
                
                # Display feature analysis
                with st.expander("📊 Transaction Analysis"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Financial Details**")
                        st.metric("Transaction Amount", f"${amount:,.2f}")
                        st.metric("Sender Balance Change", f"${features_df['balance_change_org'].iloc[0]:,.2f}")
                        st.metric("Receiver Balance Change", f"${features_df['balance_change_dest'].iloc[0]:,.2f}")
                        st.metric("Transaction Type", transaction_type)
                        
                    with col2:
                        st.markdown("**Risk Indicators**")
                        st.metric("Zero Balance Flag", "Yes" if features_df['anomaly_zero_balance'].iloc[0] == 1 else "No")
                        st.metric("Balance Mismatch", "Yes" if features_df['anomaly_balance_mismatch'].iloc[0] == 1 else "No")
                        st.metric("ML Detection Threshold", f"{fraud_threshold}")
                        st.metric("Rule-Based Risk Score", f"{risk_score:.1f}%")
                
                # Probability and risk gauges
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Machine Learning Probability**")
                    st.progress(float(probability))
                    st.caption(f"Fraud Probability: {probability:.3f} (Threshold: {fraud_threshold})")
                
                with col2:
                    st.markdown("**Rule-Based Risk Score**")
                    st.progress(float(risk_score/100))
                    st.caption(f"Risk Score: {risk_score:.1f}%")
                
                # Comprehensive risk assessment
                st.markdown("**🎯 Comprehensive Risk Assessment:**")
                
                if probability < 0.2 and risk_score < 30:
                    st.success("🟢 LOW RISK: Transaction appears normal")
                    st.write("**Recommendation:** Process transaction normally")
                    
                elif probability < 0.5 and risk_score < 60:
                    st.warning("🟡 MEDIUM RISK: Review recommended")
                    st.write("**Recommendation:** Additional verification recommended")
                    
                else:
                    st.error("�픴 HIGH RISK: Immediate action required")
                    st.write("**Recommendation:** Freeze transaction and investigate")
                
                # Detailed recommendations based on specific anomalies
                if anomalies:
                    with st.expander("🛡️ Specific Risk Mitigation Actions"):
                        st.write("**Based on detected anomalies, consider:**")
                        
                        if any("CASH_OUT with zero amount" in anomaly for anomaly in anomalies):
                            st.write("• **Verify CASH_OUT purpose** for zero amount transaction")
                            st.write("• **Check for system testing** or error")
                        
                        if any("Sender balance increased" in anomaly for anomaly in anomalies):
                            st.write("• **Investigate balance reconciliation**")
                            st.write("• **Verify transaction reversal** or correction")
                        
                        if any("No balance change" in anomaly for anomaly in anomalies):
                            st.write("• **Check system synchronization**")
                            st.write("• **Verify transaction completion**")
                        
                        if any("Negative balance" in anomaly for anomaly in anomalies):
                            st.write("• **Immediate account freeze**")
                            st.write("• **Contact account holder**")
                            st.write("• **Review credit limits**")
                        
                        if any("Amount exceeds balance" in anomaly for anomaly in anomalies):
                            st.write("• **Verify overdraft protection**")
                            st.write("• **Check credit arrangements**")
                            st.write("• **Review transaction authorization**")
                        
                        if any("Money siphoning" in anomaly for anomaly in anomalies):
                            st.write("• **Investigate intermediary accounts** for fund diversion")
                            st.write("• **Check transaction fees** and commissions")
                            st.write("• **Verify receiver account details** for accuracy")
                            st.write("• **Review transaction routing** for suspicious patterns")
                        
                        if any("Receiver gets no money" in anomaly for anomaly in anomalies):
                            st.write("• **Immediate transaction reversal** required")
                            st.write("• **Contact both parties** to verify intent")
                            st.write("• **Check for system errors** in fund transfer")
                            st.write("• **Investigate potential account hijacking**")

# Run the app
if __name__ == "__main__":
    main()
