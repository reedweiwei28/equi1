
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("⚖️ Fairness Dashboard")

np.random.seed(42)
n = 200

df = pd.DataFrame({
    "gender": np.random.choice(["Male", "Female"], n),
    "region": np.random.choice(["Urban", "Rural"], n),
    "actual": np.random.choice([0, 1], n)
})

df["predicted"] = np.where(
    (df["gender"] == "Male") & (np.random.rand(n) > 0.3), 1,
    np.where((df["gender"] == "Female") & (np.random.rand(n) > 0.6), 1, 0)
)

protected = st.selectbox("Select Protected Attribute", ["gender", "region"])

grouped = df.groupby(protected)["predicted"].mean()

st.subheader("📊 Group Metrics")
st.write(grouped)

fig, ax = plt.subplots()
grouped.plot(kind="bar", ax=ax)
ax.set_title(f"Selection Rate by {protected}")
st.pyplot(fig)

gap = grouped.max() - grouped.min()

st.subheader("⚖️ Bias Analysis")
st.write(f"Gap: {gap:.2f}")

if gap > 0.1:
    st.error("⚠️ Bias detected")
else:
    st.success("✅ No major bias")
