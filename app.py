import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

st.set_page_config(layout="wide")
st.title("⚖️ ByteTheProblem – AI Fairness Auditor")

# -------------------------
# STAGE 1: DATA INTAKE
# -------------------------
st.header("📂 Data Intake")

data_file = st.file_uploader("Upload Dataset (CSV)", type=["csv"])
pred_file = st.file_uploader("Upload Predictions (Optional CSV)", type=["csv"])

if data_file:
    df = pd.read_csv(data_file)
else:
    st.warning("Using sample dataset")
    np.random.seed(42)
    n = 300
    df = pd.DataFrame({
        "gender": np.random.choice(["Male", "Female"], n),
        "region": np.random.choice(["Urban", "Rural"], n),
        "income": np.random.randint(20000, 100000, n),
        "actual": np.random.choice([0, 1], n)
    })
    df["predicted"] = np.where(
        (df["gender"] == "Male") & (np.random.rand(n) > 0.3), 1,
        np.where((df["gender"] == "Female") & (np.random.rand(n) > 0.6), 1, 0)
    )

if pred_file:
    preds = pd.read_csv(pred_file)
    df["predicted"] = preds.iloc[:, 0]

st.subheader("Data Preview")
st.dataframe(df.head())

# -------------------------
# STAGE 2: ATTRIBUTE MAPPING
# -------------------------
st.header("⚙️ Sensitive Attribute Mapping")

protected = st.selectbox("Protected Attribute", df.columns)
target = st.selectbox("Actual Outcome", df.columns)
prediction = st.selectbox("Prediction Column", df.columns)

if protected == target or protected == prediction:
    st.error("Invalid column selection")
    st.stop()

# -------------------------
# STAGE 3: DATA AUDIT
# -------------------------
st.header("🔍 Data Audit")

col1, col2 = st.columns(2)

with col1:
    st.write("### Group Distribution")
    st.bar_chart(df[protected].value_counts())

with col2:
    st.write("### Missing Values")
    st.write(df.isnull().sum())

# proxy detection (simple correlation)
st.write("### Proxy Detection (Correlation with protected)")
numeric_df = df.select_dtypes(include=np.number)
if not numeric_df.empty:
    corr = numeric_df.corr()[target].sort_values(ascending=False)
    st.write(corr)

# -------------------------
# STAGE 4: FAIRNESS METRICS
# -------------------------
st.header("⚖️ Fairness Evaluation")

grouped = df.groupby(protected)

selection_rate = grouped[prediction].mean()

false_positive = grouped.apply(
    lambda x: ((x[prediction] == 1) & (x[target] == 0)).mean()
)

false_negative = grouped.apply(
    lambda x: ((x[prediction] == 0) & (x[target] == 1)).mean()
)

metrics = pd.DataFrame({
    "Selection Rate": selection_rate,
    "False Positive Rate": false_positive,
    "False Negative Rate": false_negative
})

st.dataframe(metrics)

# -------------------------
# STAGE 5: EXPLAINABILITY
# -------------------------
st.header("🧠 Explainability Layer")

st.write("Top correlated features (proxy indicators):")
st.write(corr.head())

st.info("⚠️ Features highly correlated with protected attribute may act as proxies.")

# -------------------------
# STAGE 6: MITIGATION (SIMULATION)
# -------------------------
st.header("🛠️ Mitigation Engine")

threshold_adjust = st.slider("Adjust Decision Threshold", -0.2, 0.2, 0.0)

df["adjusted_pred"] = np.where(
    df[prediction] + threshold_adjust > 0.5, 1, 0
)

new_selection = df.groupby(protected)["adjusted_pred"].mean()

st.write("Adjusted Selection Rate")
st.write(new_selection)

# -------------------------
# STAGE 7: TRADEOFF
# -------------------------
st.header("⚖️ Tradeoff Analysis")

original_acc = accuracy_score(df[target], df[prediction])
adjusted_acc = accuracy_score(df[target], df["adjusted_pred"])

st.write(f"Original Accuracy: {original_acc:.2f}")
st.write(f"Adjusted Accuracy: {adjusted_acc:.2f}")

# -------------------------
# STAGE 8: BIAS DETECTION
# -------------------------
st.header("🚨 Bias Detection")

gaps = metrics.max() - metrics.min()
st.write(gaps)

for m, v in gaps.items():
    if v > 0.1:
        st.error(f"Bias detected in {m}")
    else:
        st.success(f"{m} OK")

# -------------------------
# STAGE 9: REPORT
# -------------------------
st.header("📄 Report Summary")

risk = "HIGH" if gaps.max() > 0.1 else "LOW"

st.write({
    "Protected Attribute": protected,
    "Max Bias Gap": float(gaps.max()),
    "Risk Level": risk
})

# -------------------------
# STAGE 10: MONITORING (SIMULATED)
# -------------------------
st.header("📈 Monitoring (Drift Simulation)")

drift = np.random.uniform(0, 0.2)
st.write(f"Fairness Drift Score: {drift:.2f}")

if drift > 0.1:
    st.warning("⚠️ Fairness drift detected – retraining recommended")
else:
    st.success("System stable")
