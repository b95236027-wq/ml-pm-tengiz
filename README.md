# Predictive Maintenance (Oil & Gas, Kazakhstan / Tengiz)

## 1) Problem
Unplanned equipment failures in oil & gas lead to downtime and high costs.  
Goal: detect early signs of abnormal behavior using sensor data.

## 2) Solution
Binary ML classifier: **Normal vs Failure** using aggregated sensor statistics.
Output: probability of failure + recommendation (inspect / stop / continue).

## 3) Data
- Source: **3W Dataset**
- Preprocessing: convert time-series to tabular features using aggregation:
  **mean / std / min / max** per sensor window.

## 4) Model
- Baseline MVP: **Random Forest**
- Target metric: ROC-AUC

## 5) Results (MVP)
- ROC-AUC: **1.0** (MVP)
> Note: project includes strict validation to avoid leakage (group/time split + CV).

## 6) Interpretability
- Feature importance (permutation / RF importance)
- Top drivers of failure are shown in the report and demo.

## 7) Demo (Streamlit)
Run demo:
```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
