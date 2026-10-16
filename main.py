import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from datetime import datetime
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
DB_PATH = "predictions.db"
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS predictions ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "created_at TEXT, age REAL, sex TEXT, tsh REAL, t3 REAL, tt4 REAL, t4u REAL, fti REAL, "
            "on_thyroxine TEXT, thyroid_surgery TEXT, pregnant TEXT, prediction TEXT, confidence REAL)"
        )
def save_record(rec):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO predictions (created_at, age, sex, tsh, t3, tt4, t4u, fti, "
            "on_thyroxine, thyroid_surgery, pregnant, prediction, confidence) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            rec,
        )
def load_records():
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query("SELECT * FROM predictions ORDER BY created_at", conn)
init_db()
@st.cache_resource
def train_models():
    df = pd.read_csv("thyroid_dataset.csv")
    df = df[FEATURES + [COL["target"]]].replace("?", np.nan)
    for c in NUM:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()
    for c in BIN:
        df[c] = df[c].astype(str).str.strip().str.lower().isin(["male", "m", "true", "t", "1", "yes"]).astype(int)
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
ts = st.sidebar.slider("Thyroid Stimulating Hormone (TSH)", 0.0, 500.0, 1.0)
t3 = st.sidebar.slider("Triiodothyronine (T3)", 0.0, 20.0, 2.0)
tt = st.sidebar.slider("Total Thyroxine (TT4)", 0.0, 500.0, 100.0)
tu = st.sidebar.slider("Thyroxine Uptake (T4U)", 0.0, 5.0, 1.0)
ft = st.sidebar.slider("Free Thyroxine Index (FTI)", 0.0, 500.0, 100.0)
ot = st.sidebar.selectbox("On Thyroxine", ["False", "True"])
tsu = st.sidebar.selectbox("Thyroid Surgery", ["False", "True"])
pr = st.sidebar.selectbox("Pregnant", ["False", "True"])
bt = st.sidebar.button("Predict Thyroid Risk")
st.title("🫁 Thyroid Prediction App")
st.subheader("🔵 Domain: Endocrinology")
st.write("Use machine learning to estimate a patient's risk of thyroid abnormalities.")
st.write("⭐ Hosted in Streamlit")
st.write("⭐ Input Dataset Fields")
st.write("•	Age (in years) &nbsp; • Sex/Gender &nbsp; • TSH — Thyroid-Stimulating Hormone &nbsp; • T3 — Triiodo Thyronine Measurement &nbsp; • TT4 — Total Thyroxine &nbsp; • T4U — Thyroxine Uptake Measurement &nbsp; • FTI — Free Thyroxine Index &nbsp; • Pregnant — Pregnancy Indicator")
st.write(" ")
st.subheader("🔵 Technology Used")
st.write("⭐ Algorithms: Decision Tree, Random Forest, SVM (Support Vector Machine), XGBoost (Extra Gradient Boost)")
st.write("⭐ Database: SQLite")
st.write(" ")
st.subheader("🔵 Prediction Results")
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
    save_record((
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ag, sx, ts, t3, tt, tu, ft, ot, tsu, pr, str(label), float(conf),
    ))
    if str(label).lower() in ("negative", "normal", "0"):
        st.success(f"✅ Low Risk: No thyroid abnormality detected. Confidence: {conf:.1f}%")
    else:
        st.error(f"⚠️ High Risk: Predicted class is {label}. Confidence: {conf:.1f}%")
st.divider()
st.subheader("📊 Model Performance & Dataset Trends")
st.markdown("### Feature Correlation Heatmap")
corr = df.select_dtypes(include=["number"]).corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1, center=0, square=True, linewidths=1, linecolor="white", annot_kws={"size": 9}, cbar_kws={"shrink": 0.8, "label": "Correlation"}, ax=ax)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
fig.tight_layout()
st.pyplot(fig)
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
st.subheader("🔵 Patient Records Over Time")
hist = load_records()
if hist.empty:
    st.info("No records yet. Click Predict Thyroid Risk to save the first record.")
else:
    hist["created_at"] = pd.to_datetime(hist["created_at"])
    param = st.selectbox("Select parameter to visualize", ["tsh", "t3", "tt4", "t4u", "fti", "age", "confidence"])
    figh, axh = plt.subplots(figsize=(10, 4))
    axh.plot(hist["created_at"], hist[param], marker="o")
    axh.set_xlabel("Time")
    axh.set_ylabel(param.upper())
    axh.set_title(f"{param.upper()} over time")
    figh.autofmt_xdate()
    st.pyplot(figh)
    st.dataframe(hist.sort_values("created_at", ascending=False), use_container_width=True)
st.divider()
st.write(" ")
st.subheader("💻 Developed by Harish.R")
