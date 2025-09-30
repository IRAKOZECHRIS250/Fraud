# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Fraud Prediction on Mobile Money Transaction",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {font-size: 3rem; color: #1f77b4; text-align: center; margin-bottom: 2rem;}
    .sub-header {font-size: 1.5rem; color: #2e86ab; margin-bottom: 1rem;}
    .prediction-box {padding: 20px; border-radius: 10px; margin: 10px 0;}
    .fraud {background-color: #ffebee; border: 2px solid #f44336;}
    .genuine {background-color: #e8f5e8; border: 2px solid #4caf50;}
    .suspicious {background-color: #fff3e0; border: 2px solid #ff9800;}
    .risk-high {color: #f44336; font-weight: bold;}
    .risk-medium {color: #ff9800; font-weight: bold;}
    .risk-low {color: #4caf50; font-weight: bold;}
    .metric-card {background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin: 10px 0;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        return joblib.load('fraud_prediction.joblib')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

feature_columns = [
    'amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest',
    'balance_change_org','balance_change_dest','abs_balance_change_org',
    'type_CASH_IN','type_CASH_OUT','type_DEBIT','type_PAYMENT','type_TRANSFER',
    'anomaly_zero_balance','anomaly_balance_mismatch']

def detect_advanced_anomalies(transaction_data, transaction_type):
    anomalies=[]; risk_factors=[]
    amount=transaction_data['amount']
    oldbalance_org=transaction_data['oldbalanceOrg']
    newbalance_org=transaction_data['newbalanceOrig']
    oldbalance_dest=transaction_data['oldbalanceDest']
    newbalance_dest=transaction_data['newbalanceDest']
    balance_change_org=newbalance_org-oldbalance_org
    balance_change_dest=newbalance_dest-oldbalance_dest
    tol=1.0

    # Core checks
    if transaction_type=="CASH_OUT" and amount==0:
        anomalies.append("CASH_OUT transaction with zero amount"); risk_factors.append(("Zero amount CASH_OUT","HIGH"))
    if transaction_type in ["CASH_OUT","TRANSFER","PAYMENT","DEBIT"] and balance_change_org>0:
        anomalies.append(f"Sender balance increased in {transaction_type} transaction"); risk_factors.append(("Sender balance increased in debit transaction","HIGH"))
    if transaction_type=="CASH_IN" and balance_change_dest<0:
        anomalies.append("Receiver balance decreased in CASH_IN transaction"); risk_factors.append(("Receiver balance decreased in CASH_IN","HIGH"))
    if balance_change_org==0 and balance_change_dest==0 and amount>0:
        anomalies.append("No balance change despite transaction amount"); risk_factors.append(("No balance change","MEDIUM"))
    if amount>10000 and abs(balance_change_org)<amount*0.1:
        anomalies.append("Large transaction with minimal balance impact"); risk_factors.append(("Large amount, small balance change","MEDIUM"))
    if amount%1000==0 and amount>1000: risk_factors.append(("Round number amount","LOW"))
    if oldbalance_org==oldbalance_dest and newbalance_org==newbalance_dest:
        anomalies.append("Potential same account transaction"); risk_factors.append(("Same account transaction","MEDIUM"))
    if newbalance_org<0 or newbalance_dest<0:
        anomalies.append("Negative balance detected"); risk_factors.append(("Negative balance","HIGH"))
    if amount>oldbalance_org and transaction_type in ["CASH_OUT","TRANSFER","PAYMENT"]:
        anomalies.append("Transaction amount exceeds sender balance (insufficient funds)"); risk_factors.append(("Insufficient funds","HIGH"))
    if oldbalance_org==0 and newbalance_org>0 and transaction_type in ["CASH_OUT","TRANSFER"]:
        anomalies.append("Zero to positive balance in debit transaction"); risk_factors.append(("Zero to positive balance","MEDIUM"))
    if transaction_type in ["TRANSFER","CASH_OUT"]:
        expected_receiver_gain=amount; actual_receiver_gain=balance_change_dest
        if actual_receiver_gain<expected_receiver_gain*0.8 and actual_receiver_gain>0:
            loss=expected_receiver_gain-actual_receiver_gain
            anomalies.append(f"Money siphoning detected: {(loss/expected_receiver_gain)*100:.1f}% missing"); risk_factors.append(("Money siphoning","HIGH"))
        elif actual_receiver_gain<=0 and amount>0:
            anomalies.append("Receiver gets no money despite transaction"); risk_factors.append(("Receiver gets zero funds","HIGH"))

    # New conditions
    if transaction_type in ["CASH_OUT","TRANSFER","PAYMENT"] and balance_change_org>amount+tol:
        anomalies.append("Sender balance increased more than the transaction amount"); risk_factors.append(("Suspicious balance increase","HIGH"))
    if transaction_type in ["CASH_OUT","TRANSFER","PAYMENT"] and abs(balance_change_org-amount)<=tol:
        anomalies.append("Sender balance increased ≈ amount sent (expected decrease)"); risk_factors.append(("Balance increased instead of decreasing","HIGH"))
    if amount>oldbalance_org and transaction_type in ["CASH_OUT","TRANSFER","PAYMENT"] and newbalance_org>=oldbalance_org:
        anomalies.append("Insufficient funds: attempted send but sender balance not debited"); risk_factors.append(("Insufficient funds, no debit","HIGH"))
    if transaction_type in ["TRANSFER","CASH_OUT","PAYMENT"] and amount>0 and abs(balance_change_org)<=tol and balance_change_dest>=amount:
        anomalies.append("Sender balance unchanged but receiver credited"); risk_factors.append(("Non-debit with receiver credit","HIGH"))
    if transaction_type in ["TRANSFER","CASH_OUT","PAYMENT"] and amount>0:
        expected_new=oldbalance_org-amount
        if newbalance_org<expected_new-tol:
            hidden_loss=expected_new-newbalance_org
            anomalies.append(f"Sender lost {hidden_loss:.2f} more than sent"); risk_factors.append(("Excessive sender loss","HIGH"))

    return anomalies,risk_factors

def calculate_risk_score(risk_factors):
    weights={"HIGH":3,"MEDIUM":2,"LOW":1}
    total=0; max_score=0
    for _,lvl in risk_factors: total+=weights[lvl]; max_score+=weights["HIGH"]
    return (total/max_score)*100 if max_score>0 else 0

def preprocess_input(transaction_data,transaction_type):
    f={}
    f['amount']=transaction_data['amount']; f['oldbalanceOrg']=transaction_data['oldbalanceOrg']; f['newbalanceOrig']=transaction_data['newbalanceOrig']
    f['oldbalanceDest']=transaction_data['oldbalanceDest']; f['newbalanceDest']=transaction_data['newbalanceDest']
    f['balance_change_org']=f['newbalanceOrig']-f['oldbalanceOrg']; f['balance_change_dest']=f['newbalanceDest']-f['oldbalanceDest']; f['abs_balance_change_org']=abs(f['balance_change_org'])
    for t in ['CASH_IN','CASH_OUT','DEBIT','PAYMENT','TRANSFER']: f[f'type_{t}']=1 if transaction_type==t else 0
    f['anomaly_zero_balance']=1 if (f['oldbalanceOrg']==0 and f['amount']>0) else 0
    expected_new=f['oldbalanceOrg']-f['amount']
    f['anomaly_balance_mismatch']=1 if abs(f['newbalanceOrig']-expected_new)>1 else 0
    return pd.DataFrame([f])[feature_columns]

def scale_features(df):
    s=df.copy()
    for c in ['amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest']:
        s[c]=np.log1p(np.maximum(s[c],0))
    for c in s.columns:
        if s[c].std()>0: s[c]=(s[c]-s[c].mean())/s[c].std()
    return s

def predict_fraud(transaction_data,transaction_type,model):
    try:
        df=preprocess_input(transaction_data,transaction_type)
        scaled=scale_features(df)
        pred=model.predict(scaled)[0]; prob=model.predict_proba(scaled)[0][1]
        anomalies,risk_factors=detect_advanced_anomalies(transaction_data,transaction_type)
        score=calculate_risk_score(risk_factors)
        return pred,prob,df,anomalies,risk_factors,score
    except Exception as e:
        st.error(f"Error making prediction: {e}"); return None,None,None,[],[],0

def main():
    st.markdown('<div class="main-header">💰 Fraud Prediction on Mobile Money Transaction</div>',unsafe_allow_html=True)
    st.markdown("This AI-powered system detects fraudulent transactions in real-time using machine learning and anomaly rules.")
    model=load_model()
    if model is None: return

    # --- sidebar, inputs, prediction button (same as before) ---
    # OMITTED here for brevity, but keep full version from previous working app

    # In mitigation section add:
    # if any("Sender balance unchanged but receiver credited" in a for a in anomalies):
    #     st.write("• **Investigate accounting loophole** allowing credit without debit")
    #     st.write("• **Audit ledger** for unauthorized credit entries")
    # if any("Sender lost" in a for a in anomalies):
    #     st.write("• **Investigate missing funds** between debit and credit")
    #     st.write("• **Check logs** for duplicate debits")
    #     st.write("• **Escalate to fraud team** if unexplained losses persist")

if __name__=="__main__":
    main()
