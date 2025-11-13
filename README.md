---
# 🚨 Anomaly Detection Platform
---

A complete end-to-end **Anomaly Detection System** built using **Machine Learning, FastAPI, and Streamlit**.  
This project trains an Isolation Forest model, serves predictions through a REST API, and visualizes results using an interactive dashboard with real-time anomaly tracking.

---

<img width="1884" height="893" alt="Screenshot 2025-11-03 205038" src="https://github.com/user-attachments/assets/b9736a9a-3628-4035-ab37-04f4d670163a" />

---

<img width="1383" height="807" alt="Screenshot 2025-11-03 205851" src="https://github.com/user-attachments/assets/4ce34cc3-f1d6-4aee-96f2-14c2901ab099" />

---

<img width="1388" height="613" alt="Screenshot 2025-11-03 205927" src="https://github.com/user-attachments/assets/d1d98768-6f21-415a-9c14-3209a36e4bb6" />

---

<img width="1380" height="612" alt="Screenshot 2025-11-03 210011" src="https://github.com/user-attachments/assets/9d6e99a0-c8fb-4d9b-9781-025f47eed07a" />

---

## 🧠 Features

- **Machine Learning Model (Isolation Forest)** trained on synthetic data for anomaly detection  
- **FastAPI Server** for real-time predictions  
- **Prometheus Metrics Integration** to monitor API performance and prediction distribution  
- **Streamlit Dashboard** with:
  - Live or local prediction modes  
  - CSV upload and event simulation  
  - KPI metrics (Total events, anomalies, status)  
  - Interactive visualizations using Plotly  
  - Recent Alerts and threshold explanations  

---

## 🧮 Training the Model

You can train a new Isolation Forest model. This generates a file named anomaly-model.joblib which is later used by both the API and the dashboard.You can train the model using-

**python train_model.py**

---

## 🚀 Running the API Server

Start the FastAPI app to serve predictions and metrics:

API endpoints:

POST /prediction → Accepts feature vector and returns anomaly status

GET /model_information → Returns model parameters

GET /metrics → Exposes Prometheus-compatible metrics

---

## 📊 Running the Streamlit Dashboard

Launch the dashboard interface using-

**streamlit run app.py**

---

## 📈 Visualizations

🟢 Normal Data Points shown in green

🔴 Anomalies highlighted in red

🧮 Dynamic Threshold Line = Mean - Standard Deviation

📊 Bar chart, Pie chart, and recent alert tables for quick insights

---

## 📦 requirements.txt
```txt
# Core ML
scikit-learn==1.5.0
numpy==1.26.0
pandas==2.2.0
joblib==1.3.2

# API
fastapi==0.115.0
uvicorn==0.29.0
pydantic==2.7.0
prometheus-client==0.20.0

# Dashboard
streamlit==1.37.0
plotly==5.22.0
requests==2.32.0

# Optional utilities
python-dateutil==2.9.0



