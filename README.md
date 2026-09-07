# Industrial Water Quality Analysis & Prediction (ML / XGBoost)

An intelligent Industrial Water Quality Monitoring and Proactive Decision Support System built with **Streamlit**, **XGBoost**, and **Neural Networks (MLPClassifier)**.

## 📌 Features

- **Real-Time Predictive Analytics**: Monitors vital water parameters including pH, Turbidity, Temperature, and Conductivity to detect contamination risks.
- **Hybrid AI Architecture**: Combines XGBoost Classifier with Multi-Layer Perceptron (MLP) Neural Networks for high-accuracy anomaly detection and future contamination forecasting.
- **Interactive Streamlit Dashboard**: User-friendly control panel with live analytics, digital twin simulations, confusion matrix, and ROC curve visualization.
- **Industrial Logging**: Automated audit trail generation and Excel export capabilities (`industrial_log.xlsx`).

---

## 🛠️ Project Structure

```
├── app.py                  # Main Streamlit Dashboard Application
├── xgboost_water.py        # Core Machine Learning pipeline & model training
├── industrial_log.xlsx     # Generated industrial audit logs
├── water_quality.csv       # Water quality sensor dataset
├── .gitignore              # Git ignore rules for virtual environments & cache
└── README.md               # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.10+ installed.

### 2. Setup Virtual Environment
```bash
python -m venv .venv
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install streamlit xgboost scikit-learn pandas numpy matplotlib seaborn openpyxl
```

### 4. Run the Application
```bash
streamlit run app.py
```
