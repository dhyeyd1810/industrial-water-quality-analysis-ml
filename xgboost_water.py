# ============================================
# INDUSTRIAL WATER QUALITY HYBRID AI SYSTEM
# CLEAN FINAL VERSION
# ============================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from openpyxl import Workbook
from datetime import datetime
from sklearn.metrics import accuracy_score

# ============================================
# 1. GENERATE SIMULATED SENSOR DATA
# ============================================
np.random.seed(42)
num_samples = 1000 # Increased for better training

data = pd.DataFrame({
    "pH": np.random.uniform(6, 9, num_samples),
    "Turbidity": np.random.uniform(0, 5, num_samples),
    "Temperature": np.random.uniform(15, 35, num_samples),
    "Conductivity": np.random.uniform(100, 1000, num_samples),
})

# Target: 1 = Contaminated (Bad), 0 = Clean (Good)
data["target"] = (
    (data["pH"] < 6.2) | (data["pH"] > 8.8) |
    (data["Turbidity"] > 4.2) |
    (data["Conductivity"] > 850)
).astype(int)

# Future contamination (next reading)
data["future_target"] = data["target"].shift(-1)
data = data.dropna() # Remove the last row which has no future target

# Define Features and Labels
X = data[["pH", "Turbidity", "Temperature", "Conductivity"]]
y = data["target"]

# ============================================
# 2. DATA PREPARATION
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================
# 3. TRAIN HYBRID MODELS
# ============================================

# Model A: XGBoost
xgb_model = XGBClassifier(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.05,
    eval_metric="logloss"
)
xgb_model.fit(X_train, y_train)

# Model B: Neural Network (MLP)
mlp_model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    max_iter=1000,
    early_stopping=True,
    random_state=42
)
mlp_model.fit(X_train_scaled, y_train)

# Model C: Future Risk (Forecaster)
Xf_train, Xf_test, yf_train, yf_test = train_test_split(
    X, data["future_target"], test_size=0.2, random_state=42
)
future_model = XGBClassifier(eval_metric="logloss")
future_model.fit(Xf_train, yf_train)

# ============================================
# 4. SOP GENERATOR (LOGIC CORRECTED)
# ============================================
def generate_sop(prob):
    # prob 1.0 = Highly Contaminated
    # prob 0.0 = Perfectly Clean
    
    if prob < 0.3:
        return None # System is healthy

    elif 0.3 <= prob <= 0.7:
        return {
            "risk_level": "MEDIUM",
            "action_type": "Quality Warning",
            "valve_status": "Partially Closed (70%)",
            "escalation": "Supervisor Notification",
            "sop_text": f"QUALITY WARNING: Risk Probability {prob:.2%}. Reduce flow, validate sensors."
        }

    else:
        return {
            "risk_level": "CRITICAL",
            "action_type": "Emergency Shutdown",
            "valve_status": "Fully Closed",
            "escalation": "Plant Manager + Safety Officer",
            "sop_text": f"EMERGENCY: Risk Probability {prob:.2%}. Close inlet valve, trigger alarm."
        }

# ============================================
# 5. EXECUTION & LOGGING
# ============================================
xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
mlp_prob = mlp_model.predict_proba(X_test_scaled)[:, 1]
hybrid_prob = (0.6 * xgb_prob) + (0.4 * mlp_prob)

wb = Workbook()
sheet = wb.active
sheet.title = "Industrial_Log"
sheet.append(["Timestamp", "Sample ID", "Risk Prob", "Risk Level", "Action", "Valve", "Escalation", "SOP"])

warning_count = 0
critical_count = 0

for i in range(len(hybrid_prob)):
    prob = hybrid_prob[i]
    sop_data = generate_sop(prob)

    if sop_data:
        if sop_data["risk_level"] == "MEDIUM": warning_count += 1
        else: critical_count += 1
        
        sheet.append([
            datetime.now().strftime("%H:%M:%S"), i+1, round(prob, 4),
            sop_data["risk_level"], sop_data["action_type"], 
            sop_data["valve_status"], sop_data["escalation"], sop_data["sop_text"]
        ])

output_file = "industrial_log.xlsx"
wb.save(output_file)

# ============================================
# 6. OUTPUT SUMMARY
# ============================================
print(f"Model Accuracy -> XGB: {accuracy_score(y_test, xgb_model.predict(X_test)):.2%} | MLP: {accuracy_score(y_test, mlp_model.predict(X_test_scaled)):.2%}")
print("-" * 40)
print(f"Total Quality Warnings: {warning_count}")
print(f"Total Emergency Shutdowns: {critical_count}")
print(f"Excel Log Saved: {output_file}")

# Predictive Forecast for the next reading
latest_sample = X.iloc[[-1]]
future_prob = future_model.predict_proba(latest_sample)[0][1]
print("-" * 40)
print(f"PREDICTIVE FORECAST: {future_prob:.2%} risk for next cycle.")