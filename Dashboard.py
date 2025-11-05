import streamlit as st
import pandas as pd
import numpy as np
import time
import requests
from joblib import load
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ---------------------- THEME CONFIG ----------------------
OCEAN_BREEZE = {
    "primaryColor": "#00B4D8",
    "backgroundColor": "#F1FAFB",
    "secondaryBackgroundColor": "#CAF0F8",
    "textColor": "#03045E",
    "font": "sans-serif"
}
st.set_page_config(page_title="Anomaly Detection Platform",layout = "wide")

# --- Animated Blinking Siren ---
siren_html = """
<style>
@keyframes blink {
  0% {opacity: 1;}
  50% {opacity: 0.2;}
  100% {opacity: 1;}
}
.siren {
  display: inline-block;
  animation: blink 1s infinite;
}
</style>
<h1 style='text-align: center;'>
<span class='siren'>🚨</span> Anomaly Detection Platform
</h1>
"""
st.markdown(siren_html, unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)

# ---------------------- THEME STYLING ----------------------
st.markdown(f"""
    <style>
        body, .reportview-container, .main {{
            background-color: {OCEAN_BREEZE['backgroundColor']};
            color: {OCEAN_BREEZE['textColor']};
            font-family: {OCEAN_BREEZE['font']};
        }}

        /* Sidebar */
        section[data-testid="stSidebar"] {{
            background-color: {OCEAN_BREEZE['secondaryBackgroundColor']};
        }}

        /* Buttons */
        .stButton>button {{
            background-color: {OCEAN_BREEZE['primaryColor']};
            color: white;
            border-radius: 10px;
            border: none;
            padding: 0.5em 1em;
            font-weight: 600;
            transition: 0.3s;
        }}
        .stButton>button:hover {{
            background-color: #0096C7;
            transform: scale(1.03);
        }}

        /* Cards */
        .card {{
            background-color: {OCEAN_BREEZE['secondaryBackgroundColor']};
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}

        /* Metrics */
        [data-testid="stMetricValue"], [data-testid="stMetricDelta"], [data-testid="stMetricLabel"] {{
            color: {OCEAN_BREEZE['textColor']} !important;
        }}

        /* Table */
        .stDataFrame, .stTable {{
            background-color: white !important;
            color: {OCEAN_BREEZE['textColor']} !important;
            border-radius: 10px;
        }}

        /* Headings */
        h1, h2, h3 {{
            color: {OCEAN_BREEZE['primaryColor']};
        }}
    </style>
""", unsafe_allow_html=True)

# ---------------------- HELPER FUNCTIONS ----------------------
def human_status(is_inlier):
    return "Normal" if int(is_inlier) == 1 else "Anomaly"

def status_badge(is_inlier: int):
    if is_inlier is None:
        return "<div>—</div>"
    color = "green" if is_inlier == 1 else "red"
    text = "🟢 NORMAL" if is_inlier == 1 else "🔴 ANOMALY"
    badge_html = f"""
    <style>
    @keyframes blink {{
        0% {{opacity: 1;}}
        50% {{opacity: 0.4;}}
        100% {{opacity: 1;}}
    }}
    .metric-box {{
        background-color: rgba(255, 255, 255, 0.7);
        border-radius: 12px;
        padding: 14px;
        text-align: center;
        box-shadow: 0 0 8px rgba(0,0,0,0.1);
        width: 100%;
    }}
    .metric-title {{
        font-weight: 600;
        font-size: 16px;
        margin-bottom: 6px;
        color: #333;
    }}
    .blink {{
        animation: blink 1s infinite;
        font-size: 22px;
        font-weight: bold;
        color: {color};
    }}
    </style>
    <div class="metric-box">
        <div class="metric-title">Latest Status</div>
        <div class="blink">{text}</div>
    </div>
    """
    return badge_html

def post_to_api(endpoint: str, feature_vector):
    payload = {"feature_vector": feature_vector, "score": True}
    try:
        r = requests.post(endpoint, json=payload, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def local_predict(model, feature_vector):
    arr = np.array([feature_vector])
    pred = model.predict(arr)[0]
    try:
        score = model.score_samples(arr)[0]
    except Exception:
        score = float("nan")
    return {"is_inlier": int(pred), "anomaly_score": float(score)}

# ---------------------- SIDEBAR CONTROLS ----------------------
st.sidebar.title("⚙️**Controls**")
mode = st.sidebar.selectbox("**Mode**", ["Local model", "Live API"])
if mode == "Live API":
    api_url = st.sidebar.text_input("**Prediction API URL**", "http://127.0.0.1:8000/prediction")
else:
    api_url = None

st.sidebar.markdown("---")
data_source = st.sidebar.radio("**Input Source**", ["Simulate events", "Upload CSV"])
simulate_count = st.sidebar.slider("**Batch simulate N events**", 1, 200, 10)
auto_seed = st.sidebar.checkbox("**Randomize seed each run**", value=True)

st.sidebar.markdown("""
_______________________________________________
**🧠Models used**
_______________________________________________
**Isolation Forest (Unsupervised tree ensemble)**
**Definition:** Used for fast tabular anomaly detection. 
**Why it's used:** Isolates anomalies by random partitioning.
_______________________________________________
**Autoencoder (Neural network)**
**Definition:** Useful for nonlinear patterns  
**Why it's used:** Flags high reconstruction error as anomaly.
_______________________________________________
**One-Class SVM (Kernel method)**  
**Definition:** Defines boundary of normal data  
**Why it's used:** Good for smaller datasets.
_______________________________________________  
**LOF (Local Outlier Factor) (Density-based)**
**Definition:** Detects points with lower local density compared to neighbors  
**Why it's used:** Useful for local anomalies.
_______________________________________________
""")

# ---------------------- LOAD LOCAL MODEL ----------------------
local_model = None
if mode == "Local model":
    model_path = st.sidebar.text_input("Local model path (joblib)", "anomaly-model.joblib")
    try:
        local_model = load(model_path)
        st.sidebar.success("**Local model loaded** ✅")
    except Exception:
        st.sidebar.warning("Model not found or failed to load.")

# ---------------------- DATA STATE ----------------------
if "events" not in st.session_state:
    st.session_state.events = pd.DataFrame(columns=[
        "timestamp", "user", "resource", "action", "feature_0", "feature_1", "is_inlier", "anomaly_score"
    ])

# ---------------------- DATA INPUT (UPLOAD/SIMULATE) ----------------------
uploaded = None
if data_source == "Upload CSV":
    uploaded = st.file_uploader("Upload CSV of events (columns: user,resource,action,feature_0,feature_1)", type=["csv"])
    if uploaded is not None:
        df_up = pd.read_csv(uploaded)
        expected = {"user", "resource", "action", "feature_0", "feature_1"}
        if not expected.issubset(set(df_up.columns)):
            st.error("CSV missing required columns. Required: user, resource, action, feature_0, feature_1")
        else:
            st.success(f"Loaded {len(df_up)} rows from CSV")
            # Predict for rows and append
            new_rows = []
            for _, r in df_up.iterrows():
                fv = [float(r["feature_0"]), float(r["feature_1"])]
                if mode == "Live API":
                    res = post_to_api(api_url, fv)
                    if "error" in res:
                        st.error("API error: " + res["error"])
                        break
                else:
                    if local_model is None:
                        st.warning("No local model; skipping prediction for upload.")
                        break
                    res = local_predict(local_model, fv)
                new_rows.append({
                    "timestamp": datetime.utcnow(),
                    "user": r["user"],
                    "resource": r["resource"],
                    "action": r["action"],
                    "feature_0": fv[0],
                    "feature_1": fv[1],
                    "is_inlier": res.get("is_inlier", None),
                    "anomaly_score": res.get("anomaly_score", None)
                })
            if new_rows:
                st.session_state.events = pd.concat([st.session_state.events, pd.DataFrame(new_rows)], ignore_index=True)

# ---------------------- SIMULATION BUTTONS ----------------------
col1, col2, col3 = st.columns([1,1,1])
with col1:
    if st.button("🎲 Simulate Next Event"):
        seed = None if auto_seed else 42 + len(st.session_state.events)
        if seed: np.random.seed(seed)
        u = f"user_{np.random.randint(1,8)}"
        rsc = f"dataset_{np.random.randint(1,10)}"
        act = np.random.choice(["read", "write", "export"])
        f0, f1 = np.random.normal(0, 1, 2)
        fv = [f0, f1]
        if mode == "Live API":
            res = post_to_api(api_url, fv)
        else:
            res = local_predict(local_model, fv) if local_model else {"is_inlier": None, "anomaly_score": None}
        new = {
            "timestamp": datetime.utcnow(),
            "user": u,
            "resource": rsc,
            "action": act,
            "feature_0": f0,
            "feature_1": f1,
            "is_inlier": res.get("is_inlier"),
            "anomaly_score": res.get("anomaly_score")
        }
        st.session_state.events = pd.concat([st.session_state.events, pd.DataFrame([new])], ignore_index=True)
with col2:
    if st.button(f"🚀 Simulate {simulate_count} Events"):
        for _ in range(simulate_count):
            f0, f1 = np.random.normal(0, 1, 2)
            fv = [f0, f1]
            if mode == "Live API":
                res = post_to_api(api_url, fv)
            else:
                res = local_predict(local_model, fv) if local_model else {"is_inlier": None, "anomaly_score": None}
            new = {
                "timestamp": datetime.utcnow(),
                "user": f"user_{np.random.randint(1,8)}",
                "resource": f"dataset_{np.random.randint(1,10)}",
                "action": np.random.choice(["read", "write", "export"]),
                "feature_0": f0,
                "feature_1": f1,
                "is_inlier": res.get("is_inlier"),
                "anomaly_score": res.get("anomaly_score")
            }
            st.session_state.events = pd.concat([st.session_state.events, pd.DataFrame([new])], ignore_index=True)
with col3:
    if st.button("🧹 Clear Events"):
        st.session_state.events = st.session_state.events.iloc[0:0]
        st.success("Events cleared ✅")

# ---------------------- METRICS ----------------------
events = st.session_state.events.copy()
total = len(events)
anomalies = events[events["is_inlier"] != 1]
num_anomalies = anomalies.shape[0] if total > 0 else 0
pct_anomalies = (num_anomalies / total * 100) if total > 0 else 0

# KPI layout
kpi1, kpi2, kpi3 = st.columns(3)
with kpi1:
    st.metric("**Total Events**", total)

with kpi2:
    st.metric("**Anomalies**", f"{num_anomalies}", delta=f"{pct_anomalies:.1f}%")

with kpi3:
    latest_inlier = events["is_inlier"].iloc[-1] if total > 0 else None
    st.markdown(status_badge(latest_inlier), unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)


# ---------------------- 📊 CHART 1: Anomaly Score Over Time ----------------------
st.subheader("📊**Anomaly Score Over Time**")

if total > 0:
    # Compute threshold dynamically
    threshold = events["anomaly_score"].mean() - events["anomaly_score"].std()

    # Separate normal and anomaly points
    normal_points = events[events["anomaly_score"] >= threshold]
    anomaly_points = events[events["anomaly_score"] < threshold]

    # Base line chart
    fig = px.line(events, x="timestamp", y="anomaly_score", title="")

    # Add normal points (green)
    fig.add_scatter(
        x=normal_points["timestamp"],
        y=normal_points["anomaly_score"],
        mode="markers",
        name="Normal",
        marker=dict(color="green", size=8),
    )

    # Add anomaly points (red)
    fig.add_scatter(
        x=anomaly_points["timestamp"],
        y=anomaly_points["anomaly_score"],
        mode="markers",
        name="Anomaly",
        marker=dict(color="red", size=8),
    )

    # Add threshold line
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="orange",
        annotation_text=f"Threshold (mean - std = {threshold:.4f})",
        annotation_position="top left"
    )

    # Style and layout
    fig.update_traces(line_color=OCEAN_BREEZE["primaryColor"], selector=dict(mode="lines"))
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis_title="Anomaly score",
        xaxis_title="Time",
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Info line below chart ---
    st.markdown(f"""
            <div style="background-color:#E0F7FA;padding:12px;border-radius:10px;margin-top:-10px;
                        box-shadow:0 2px 6px rgba(0,0,0,0.1);">
                <p style="font-size:16px; font-weight:600; color:#0077B6;">
                    🧮Threshold Calculation: <span style="color:#03045E;">
                    Threshold = Mean - Standard Deviation = {threshold:.4f}</span>
                </p>
                <p style="font-size:15px; color:#023E8A;">
                    🟢 <b>Normal Score</b> — If the score ≥ {threshold:.4f}<br>
                    🔴 <b>Anomaly Score</b> — If the score &lt; {threshold:.4f}
                </p>
            </div>
        """, unsafe_allow_html=True)
else:
    st.info("No events yet.")
st.markdown('<div class="card">', unsafe_allow_html=True)

# ---------------------- 🔢 CHART 2: Activity Trends by User ----------------------
st.subheader("👤**Activity Trends by User**")
if total > 0:
    top_users = events.tail(100).groupby("user").size().reset_index(name="count")
    fig2 = px.bar(top_users, x="user", y="count", color="user", title="")
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("No activity yet.")

# ---------------------- 🚨 CHART 3: % of Anomalies ----------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("**% of Normal🟢 and Anomalies🔴**")
if total > 0:
    labels = ["Normal", "Anomaly"]
    values = [total - num_anomalies, num_anomalies]
    colors = ["#2ECC71", "#E74C3C"]
    fig3 = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.5)])
    fig3.update_traces(marker=dict(colors=colors, line=dict(color='white', width=2)))
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("No anomalies yet.")

# ---------------------- 🗂️ CHART 4: Recent Alerts ----------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("**🚨Recent Alerts**")
if total > 0:
    recent = events.sort_values("timestamp", ascending=False).head(10)
    recent["status"] = recent["is_inlier"].apply(lambda v: "🟢 Normal" if int(v) == 1 else "🔴 Anomaly")
    st.table(recent[["timestamp", "user", "resource", "action", "anomaly_score", "status"]])
else:
    st.info("No alerts to display.")

# ---------- Detailed view ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("**🧾Events log**")
st.dataframe(events.sort_values("timestamp", ascending=False).reset_index(drop=True))
st.markdown('<div class="card">', unsafe_allow_html=True)
