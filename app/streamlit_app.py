import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
import pandas as pd
import streamlit as st
import joblib
import os

from src.config import SAMPLE_PATH, MODEL_PATH, TARGET_COL

st.set_page_config(page_title="Predictive Maintenance Demo", layout="centered")

st.title("Predictive Maintenance — Oil & Gas (Demo)")
st.write("Binary classification: **Normal vs Failure** using aggregated sensor features.")

# Load sample
df = pd.read_csv(SAMPLE_PATH)

st.subheader("Sample data")
st.dataframe(df.head(10), use_container_width=True)

# Load model
if not os.path.exists(MODEL_PATH):
    st.error(f"Model not found: {MODEL_PATH}. Run training first: `python -m src.train`")
    st.stop()

model = joblib.load(MODEL_PATH)

# Choose row
st.subheader("Make a prediction")
row_idx = st.number_input("Row index", min_value=0, max_value=len(df)-1, value=0, step=1)

row = df.iloc[[row_idx]].copy()

if TARGET_COL in row.columns:
    y_true = int(row[TARGET_COL].iloc[0])
    row_features = row.drop(columns=[TARGET_COL])
else:
    y_true = None
    row_features = row

proba = float(model.predict_proba(row_features)[:, 1][0])

st.write(f"**Failure probability:** `{proba:.3f}`")

# Simple recommendation rule
if proba >= 0.7:
    rec = "🟥 STOP / Immediate inspection"
elif proba >= 0.4:
    rec = "🟧 Inspect soon"
else:
    rec = "🟩 Continue operation"

st.write(f"**Recommendation:** {rec}")

if y_true is not None:
    st.write(f"**True label (from sample):** `{y_true}`")
