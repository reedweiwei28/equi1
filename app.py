import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="AI Fairness Auditor", layout="wide")

st.title("⚖️ AI Fairness Auditor Dashboard")

st.info("""
📌 Upload a CSV file with:
- Protected attribute (e.g., gender, caste)
- Actual outcome column (0/1)
- Model prediction column (0/1)
""")

# -------------------------
# FILE UPLOAD
# -------------------------
uploaded_file = st.file_uploader("📂 Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully")

    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

else:
    st.warning("⚠️ No file uploaded. Using sample dataset.")

    np.random.seed(42)
    n = 300

    df = pd.DataFrame({
        "gender": np.random.choice(["Male", "Female"], n),
        "region": np.random.choice(["Urban", "Rural"], n),
        "actual": np.random.choice([0, 1], n)
    })

    df["predicted"] = np.where(
        (df["gender"] == "Male") & (np.random.rand(n) > 0.3), 1,
        np.where((df["gender"] == "Female") & (np.random.rand(n) > 0.6), 1, 0)
    )

# -------------------------
# SIDEBAR CONFIG
# -------------------------
st.sidebar.header("⚙️ Configuration")

protected = st.sidebar.selectbox("Protected Attribute", df.columns)
target = st.sidebar.selectbox("Actual Outcome (y_true)", df.columns)
prediction = st.sidebar.selectbox("Model Prediction (y_pred)", df.columns)

# validation
if protected == target or protected == prediction:
    st.error("❌ Protected attribute cannot be same as target or prediction")
    st.stop()

# -------------------------
# DATA AUDIT
# -------------------------
st.subheader("🔍 Data Audit")

col1, col2 = st.columns(2)

with col1:
    st.write("### Group Distribution")
    st.bar_chart(df[protected].value_counts())

with col2:
    st.write("### Missing Values")
    st.write(df.isnull().sum())

# -------------------------
# FAIRNESS METRICS
# -------------------------
st.subheader("⚖️ Fairness Metrics")

grouped = df.groupby(protected)

selection_rate = grouped[prediction].mean()

false_positive = grouped.apply(
    lambda x: ((x[prediction] == 1) & (x[target] == 0)).mean()
)

false_negative = grouped.apply(
    lambda x: ((x[prediction] == 0) & (x[target] == 1)).mean()
)

metrics_df = pd.DataFrame({
    "Selection Rate": selection_rate,
    "False Positive Rate": false_positive,
    "False Negative Rate": false_negative
})

st.dataframe(metrics_df)

# -------------------------
# VISUALIZATION
# -------------------------
st.subheader("📊 Visual Comparison")

fig, ax = plt.subplots()
metrics_df.plot(kind="bar", ax=ax)
ax.set_title(f"Fairness Metrics by {protected}")
st.pyplot(fig)

# -------------------------
# BIAS DETECTION
# -------------------------
st.subheader("🚨 Bias Detection")

gaps = metrics_df.max() - metrics_df.min()

st.write("### Metric Gaps")
st.write(gaps)

threshold = 0.1

for metric, value in gaps.items():
    if value > threshold:
        st.error(f"⚠️ Bias detected in {metric} (gap = {value:.2f})")
    else:
        st.success(f"✅ {metric} within acceptable range")

# -------------------------
# MODEL PERFORMANCE
# -------------------------
st.subheader("📈 Model Performance")

try:
    acc = accuracy_score(df[target], df[prediction])
    st.write(f"Accuracy: {acc:.2f}")
except:
    st.warning("⚠️ Could not compute accuracy (check data format)")

# -------------------------
# INSIGHTS
# -------------------------
st.subheader("🧠 Insights")

if gaps["Selection Rate"] > 0.1:
    st.write("The model favors one group significantly in approvals.")

if gaps["False Positive Rate"] > 0.1:
    st.write("Certain groups are more likely to be incorrectly approved.")

if gaps["False Negative Rate"] > 0.1:
    st.write("Certain groups are more likely to be unfairly rejected.")

st.write("Recommendation: Review training data and consider fairness-aware adjustments.")

# -------------------------
# SUMMARY REPORT
# -------------------------
st.subheader("📄 Audit Summary")

risk = "High" if gaps.max() > 0.1 else "Low"

st.write({
    "Protected Attribute": protected,
    "Highest Bias Gap": float(gaps.max()),
    "Risk Level": risk
})
