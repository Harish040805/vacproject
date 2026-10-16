import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
st.set_page_config(page_title="Thyroid Prediction App", page_icon="🫁", layout="wide")
COL = {
    "age": "age",
    "sex": "sex",
    "tsh": "TSH",
    "t3": "T3",
    "tt4": "TT4",
    "t4u": "T4U",
    "fti": "FTI",
    "thyroxine": "on_thyroxine",
    "surgery": "thyroid_surgery",
    "pregnant": "pregnant",
    "target": "thyroid_class",
}
NUM = [COL[k] for k in ["age", "tsh", "t3", "tt4", "t4u", "fti"]]
BIN = {COL["sex"]: "Male", COL["thyroxine"]: "True", COL["surgery"]: "True", COL["pregnant"]: "True"}
FEATURES = NUM + list(BIN)
@st.cache_resource
def train_models():
    df = pd.read_csv("thyroid_dataset.csv")
    df = df[FEATURES + [COL["target"]]].replace("?", np.nan)
    for c in NUM:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()
    for c, v in BIN.items():
        df[c] = (df[c] == v).astype(int)
    le = LabelEncoder()
    y = le.fit_transform(df[COL["target"]])
    X = df[FEATURES]
    strat = y if np.bincount(y).min() > 1 else None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=strat)
    dt = DecisionTreeClassifier(random_state=42).fit(X_tr, y_tr)
    gb = GradientBoostingClassifier(random_state=42).fit(X_tr, y_tr)
    acc = {
        "Decision Tree": accuracy_score(y_te, dt.predict(X_te)) * 100,
        "Gradient Boosting": accuracy_score(y_te, gb.predict(X_te)) * 100,
    }
    return df, le, gb, acc
df, le, gb, acc = train_models()
st.sidebar.header("Patient Input Features")
ag = st.sidebar.slider("Patient Age", 1, 120, 30)
sx = st.sidebar.selectbox("Sex", ["Male", "Female"])
ts = st.sidebar.slider("TSH", 0.0, 500.0, 1.0)
t3 = st.sidebar.slider("T3", 0.0, 20.0, 2.0)
tt = st.sidebar.slider("TT4", 0.0, 500.0, 100.0)
tu = st.sidebar.slider("T4U", 0.0, 5.0, 1.0)
ft = st.sidebar.slider("FTI", 0.0, 500.0, 100.0)
ot = st.sidebar.selectbox("On Thyroxine", ["False", "True"])
tsu = st.sidebar.selectbox("Thyroid Surgery", ["False", "True"])
pr = st.sidebar.selectbox("Pregnant", ["False", "True"])
bt = st.sidebar.button("Predict Thyroid Risk")
st.title("🫁 Thyroid Prediction App")
st.write("Use machine learning to estimate a patient's risk of thyroid abnormalities.")
st.subheader("Prediction Results")
if bt:
    vals = {
        COL["age"]: ag,
        COL["tsh"]: ts,
        COL["t3"]: t3,
        COL["tt4"]: tt,
        COL["t4u"]: tu,
        COL["fti"]: ft,
        COL["sex"]: int(sx == "Male"),
        COL["thyroxine"]: int(ot == "True"),
        COL["surgery"]: int(tsu == "True"),
        COL["pregnant"]: int(pr == "True"),
    }
    row = pd.DataFrame([vals])[FEATURES]
    pred = gb.predict(row)[0]
    label = le.inverse_transform([pred])[0]
    conf = gb.predict_proba(row)[0].max() * 100
    if str(label).lower() in ("negative", "normal", "0"):
        st.success(f"Low Risk: No thyroid abnormality detected. Confidence: {conf:.1f}%")
    else:
        st.error(f"High Risk: Predicted class is {label}. Confidence: {conf:.1f}%")
st.divider()
st.subheader("📊 Model Performance & Dataset Trends")
m1, m2 = st.columns(2)
m1.metric("Decision Tree Accuracy", f"{acc['Decision Tree']:.1f}%")
m2.metric("Gradient Boosting Accuracy", f"{acc['Gradient Boosting']:.1f}%")
c1, c2 = st.columns(2)
fig1, ax1 = plt.subplots()
ax1.bar(list(acc.keys()), list(acc.values()), color=["#92c5de", "#e8867c"])
ax1.set_ylim(0, 100)
ax1.set_ylabel("Accuracy (%)")
ax1.set_title("Model Accuracy Comparison")
c1.pyplot(fig1)
fig2, ax2 = plt.subplots()
sample = df.sample(min(300, len(df)), random_state=42)
colors = ["#d7191c", "#2c7bb6", "#1a9641", "#ff7f00"]
markers = ["o", "x", "s", "^"]
for i, cls in enumerate(sample[COL["target"]].unique()):
    sub = sample[sample[COL["target"]] == cls]
    ax2.scatter(sub[COL["age"]], sub[COL["tsh"]], c=colors[i % 4], marker=markers[i % 4], label=str(cls))
ax2.set_xlabel("Age")
ax2.set_ylabel("TSH")
ax2.set_title("Age vs. TSH")
ax2.legend(title="Thyroid Class")
c2.pyplot(fig2)
st.divider()
st.write(" ")
st.subheader("💻 Developed by Harish.R")
