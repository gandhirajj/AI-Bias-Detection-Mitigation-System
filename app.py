# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate, false_positive_rate
import shap
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Bias Detection & Mitigation Prototype")

st.title("AI Bias Detection & Simple Mitigation (Hackathon Prototype)")

# ---------- Upload / Load ----------
uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])
sample_data_button = st.button("Load demo dataset (PIMA diabetes)")

if sample_data_button:
    from sklearn.datasets import load_diabetes
    # Using sklearn diabetes is regression; instead let's load the PIMA from URL if internet unavailable.
    # For the demo, create a synthetic dataset (binary target) to demonstrate flow.
    n = 1000
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "age": rng.randint(21, 85, size=n),
        "bmi": rng.normal(27, 6, size=n),
        "glucose": rng.normal(120, 30, size=n),
        "gender": rng.choice(["Male", "Female"], size=n, p=[0.7, 0.3]),
    })
    # synthetic target correlated with glucose and gender bias
    df["target"] = ((df["glucose"] + (df["gender"] == "Male")*5 + rng.normal(0,20,n)) > 130).astype(int)
    data = df
    st.success("Loaded synthetic demo dataset.")
elif uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.success("Dataset loaded.")
else:
    st.info("Upload a CSV or click 'Load demo dataset' to try the app.")
    st.stop()

st.subheader("Dataset preview")
st.dataframe(data.head())

# ---------- User selects features ----------
all_columns = data.columns.tolist()
col1, col2 = st.columns(2)
with col1:
    target_col = st.selectbox("Select target (binary)", options=all_columns, index=len(all_columns)-1)
with col2:
    sensitive_col = st.selectbox("Select sensitive attribute (e.g., gender, race)", options=all_columns, index=0)

# Auto choose numeric features except target and sensitive
feature_cols = [c for c in all_columns if c not in [target_col, sensitive_col] and pd.api.types.is_numeric_dtype(data[c])]
st.write("Auto-selected numeric features (you can modify):")
feature_cols = st.multiselect("Features", options=feature_cols, default=feature_cols)

if len(feature_cols) == 0:
    st.error("Pick at least one numeric feature to train.")
    st.stop()

# ---------- Prepare data ----------
X = data[feature_cols].copy()
y = data[target_col].astype(int)
A = data[sensitive_col].astype(str)  # protected attribute

# Basic preprocessing: fill na, scale numeric
X = X.fillna(X.median())
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Train/test split (stratify by y if possible)
X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
    X_scaled, y, A, test_size=0.25, random_state=42, stratify=y if len(np.unique(y))>1 else None
)

# ---------- Train baseline models ----------
st.subheader("Train models")
model_choice = st.multiselect("Choose models to train", options=["LogisticRegression", "RandomForest"], default=["LogisticRegression"])

models = {}
if "LogisticRegression" in model_choice:
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    models["LogisticRegression"] = lr

if "RandomForest" in model_choice:
    rf = RandomForestClassifier(n_estimators=150, random_state=42)
    rf.fit(X_train, y_train)
    models["RandomForest"] = rf

if not models:
    st.error("Select at least one model to train.")
    st.stop()

# ---------- Evaluation function ----------
def compute_metrics(est, X_t, y_t, A_t):
    y_pred = est.predict(X_t)
    y_prob = est.predict_proba(X_t)[:,1] if hasattr(est, "predict_proba") else None

    overall = {
        "accuracy": accuracy_score(y_t, y_pred),
        "precision": precision_score(y_t, y_pred, zero_division=0),
        "recall": recall_score(y_t, y_pred, zero_division=0),
        "f1": f1_score(y_t, y_pred, zero_division=0),
    }
    if y_prob is not None:
        overall["roc_auc"] = roc_auc_score(y_t, y_prob)

    # Fairness metrics by group using fairlearn MetricFrame
    metrics = {
        "accuracy": accuracy_score,
        "selection_rate": selection_rate,
        "tpr": true_positive_rate,
        "fpr": false_positive_rate
    }
    mf = MetricFrame(metrics=metrics, y_true=y_t, y_pred=y_pred, sensitive_features=A_t)
    group_metrics = mf.by_group

    # Disparate Impact: selection_rate ratio (unpriv / priv). Choose privileged as the group with highest selection rate.
    sel_rates = group_metrics["selection_rate"]
    privileged = sel_rates.idxmax()
    unprivileged = sel_rates.idxmin()
    di = sel_rates.loc[unprivileged] / sel_rates.loc[privileged] if sel_rates.loc[privileged] > 0 else np.nan

    return overall, group_metrics, {"privileged": privileged, "unprivileged": unprivileged, "disparate_impact": di}

# ---------- Show baseline results ----------
st.subheader("Baseline evaluation (Before mitigation)")
results_before = {}
for name, m in models.items():
    overall, group_metrics, di_info = compute_metrics(m, X_test, y_test, A_test)
    results_before[name] = {"overall": overall, "by_group": group_metrics, "di": di_info}
    st.markdown(f"### {name}")
    st.write("**Overall metrics**")
    st.json(overall)
    st.write("**Group metrics (by sensitive attribute)**")
    st.dataframe(group_metrics)
    st.write(f"Disparate Impact (min_group / max_group): {di_info['disparate_impact']:.3f}  (privileged: {di_info['privileged']}, unprivileged: {di_info['unprivileged']})")

# ---------- Simple reweighing mitigation ----------
st.subheader("Simple Reweighing Mitigation (pre-processing)")
st.write("""
This is a **simple reweighing**: we compute weight per protected-group so minority groups get higher sample weight.
Weights are then passed to model.fit(sample_weight=...) where supported.
""")
if st.button("Apply reweighing and retrain"):
    # compute group frequencies in training set
    grp_counts = A_train.value_counts(normalize=True)
    # weight = inverse of group frequency -> makes rare groups count more
    group_weights = {g: 1.0/(freq + 1e-6) for g, freq in grp_counts.items()}

    sample_weights = A_train.map(group_weights).astype(float)

    models_after = {}
    if "LogisticRegression" in model_choice:
        lr2 = LogisticRegression(max_iter=1000)
        lr2.fit(X_train, y_train, sample_weight=sample_weights)
        models_after["LogisticRegression"] = lr2

    if "RandomForest" in model_choice:
        rf2 = RandomForestClassifier(n_estimators=150, random_state=42)
        rf2.fit(X_train, y_train, sample_weight=sample_weights)
        models_after["RandomForest"] = rf2

    st.success("Retrained models with group reweighing.")
    # Evaluate after
    st.subheader("Evaluation after Reweighing")
    results_after = {}
    for name, m in models_after.items():
        overall, group_metrics, di_info = compute_metrics(m, X_test, y_test, A_test)
        results_after[name] = {"overall": overall, "by_group": group_metrics, "di": di_info}
        st.markdown(f"### {name} (after reweighing)")
        st.write("**Overall metrics**")
        st.json(overall)
        st.write("**Group metrics (by sensitive attribute)**")
        st.dataframe(group_metrics)
        st.write(f"Disparate Impact (min_group / max_group): {di_info['disparate_impact']:.3f}  (privileged: {di_info['privileged']}, unprivileged: {di_info['unprivileged']})")

    # Show before vs after comparison for one chosen model
    compare_model = st.selectbox("Choose a model to compare before vs after", options=list(models_after.keys()))
    if compare_model:
        bef = results_before[compare_model]["by_group"]
        aft = results_after[compare_model]["by_group"]
        st.write("TPR before vs after")
        df_cmp = pd.DataFrame({
            "tpr_before": bef["tpr"],
            "tpr_after": aft["tpr"]
        })
        st.bar_chart(df_cmp)

        st.write("Selection rate (positive rate) before vs after")
        df_cmp2 = pd.DataFrame({
            "sel_before": bef["selection_rate"],
            "sel_after": aft["selection_rate"]
        })
        st.bar_chart(df_cmp2)

    # assign for later explainability use
    models = models_after
else:
    st.info("Click the button to apply reweighing and retrain models (in-memory).")

# ---------- Explainability with SHAP ----------
st.subheader("Model Explainability (SHAP)")

chosen_for_shap = st.selectbox("Choose model for SHAP explanation", options=list(models.keys()))
if chosen_for_shap:
    model_shap = models[chosen_for_shap]
    # shap requires background dataset - take small sample from train
    try:
        explainer = shap.Explainer(model_shap.predict_proba if hasattr(model_shap, "predict_proba") else model_shap.predict, X_train)
        shap_values = explainer(X_test)
        st.write("SHAP summary plot (global feature importance)")
        fig = shap.plots.bar(shap_values, show=False)
        st.pyplot(bbox_inches="tight")
    except Exception as e:
        st.error(f"SHAP failed: {e}. Try a tree model or ensure shap is installed and model supports predict_proba/predict.")

# ---------- Utility: Export report ----------
st.subheader("Export quick CSV report")
if st.button("Download fairness metrics CSV"):
    rows = []
    for name, r in results_before.items():
        for grp, vals in r["by_group"].iterrows():
            rows.append({
                "model": name,
                "group": grp,
                "selection_rate": vals["selection_rate"],
                "tpr": vals["tpr"],
                "fpr": vals["fpr"]
            })
    df_report = pd.DataFrame(rows)
    csv = df_report.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, file_name="fairness_report.csv", mime="text/csv")

st.write("Prototype complete. For production use: add proper missing-value handling, categorical encoding, stronger mitigation algorithms (aif360 reweighing, Fairlearn reductions, adversarial debiasing), logging, tests, and secure model serving.")
