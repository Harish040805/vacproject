import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import sqlite3
import streamlit as st
from datetime import datetime
from zoneinfo import ZoneInfo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
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
class Ensemble:
    def __init__(self, models):
        self.models = models
    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.models], axis=0)
    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)
@st.cache_resource
def train_models():
    df = pd.read_csv("thyroid_dataset.csv")
    df = df[FEATURES + [COL["target"]]].replace("?", np.nan)
    for c in NUM:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()
    df = df[df[COL["age"]].between(1, 120) & (df[COL["t4u"]] > 0)]
    for c in BIN:
        df[c] = df[c].astype(str).str.strip().str.lower().isin(["male", "m", "true", "t", "1", "yes"]).astype(int)
    df = df.drop_duplicates().reset_index(drop=True)
    le = LabelEncoder()
    y = le.fit_transform(df[COL["target"]])
    X = df[FEATURES]
    strat = y if np.bincount(y).min() > 1 else None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=strat)
    sw = compute_sample_weight("balanced", y_tr)
    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1),
        "SVM": make_pipeline(StandardScaler(), SVC(probability=True, class_weight="balanced", random_state=42)),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1, eval_metric="mlogloss", random_state=42),
    }
    fitted = {}
    for n, m in models.items():
        if n == "XGBoost":
            m.fit(X_tr, y_tr, sample_weight=sw)
        else:
            m.fit(X_tr, y_tr)
        fitted[n] = m
    fitted["Ensemble"] = Ensemble(list(fitted.values()))
    acc = {n: accuracy_score(y_te, m.predict(X_te)) * 100 for n, m in fitted.items()}
    f1 = {n: f1_score(y_te, m.predict(X_te), average="macro") * 100 for n, m in fitted.items()}
    return df, le, fitted, acc, f1
df, le, fitted, acc, f1 = train_models()
def meter(val, low, high):
    if val < low:
        color = "#2c7bb6"
    elif val > high:
        color = "#d7191c"
    else:
        color = "#1a9641"
    pct = min(max(val / (high * 2) * 100, 0), 100)
    st.sidebar.markdown(
        f'<div style="background:#e0e0e0;border-radius:6px;height:8px;margin:-8px 0 14px 0;">'
        f'<div style="width:{pct}%;background:{color};height:8px;border-radius:6px;"></div></div>',
        unsafe_allow_html=True,
    )
model_name = st.sidebar.selectbox("Algorithm", list(fitted), index=len(fitted) - 1)
mdl = fitted[model_name]
rg = {c: (float(df[c].min()), float(df[c].max()), float(df[c].median())) for c in NUM}
nr = df[df[COL["target"]] == "normal"][["TSH", "T3", "TT4", "T4U", "FTI"]]
ref = {k: (float(nr[k].quantile(0.05)), float(nr[k].quantile(0.95))) for k in nr.columns}
st.sidebar.header("Patient Input Features")
ag = st.sidebar.slider("Patient Age", int(rg["age"][0]), int(rg["age"][1]), int(rg["age"][2]))
sx = st.sidebar.selectbox("Gender", ["Male", "Female"])
ts = st.sidebar.slider("Thyroid Stimulating Hormone (TSH)", rg["TSH"][0], rg["TSH"][1], rg["TSH"][2], step=0.1)
meter(ts, *ref["TSH"])
t3 = st.sidebar.slider("Triiodothyronine (T3)", rg["T3"][0], rg["T3"][1], rg["T3"][2], step=0.1)
meter(t3, *ref["T3"])
tt = st.sidebar.slider("Total Thyroxine (TT4)", rg["TT4"][0], rg["TT4"][1], rg["TT4"][2], step=1.0)
meter(tt, *ref["TT4"])
tu = st.sidebar.slider("Thyroxine Uptake (T4U)", rg["T4U"][0], rg["T4U"][1], rg["T4U"][2], step=0.01)
meter(tu, *ref["T4U"])
ft = st.sidebar.slider("Free Thyroxine Index (FTI)", rg["FTI"][0], rg["FTI"][1], rg["FTI"][2], step=0.1)
meter(ft, *ref["FTI"])
ot = st.sidebar.selectbox("On Thyroxine", ["False", "True"])
tsu = st.sidebar.selectbox("Thyroid Surgery", ["False", "True"])
pr = st.sidebar.selectbox("Pregnant", ["False", "True"], disabled=(sx == "Male"))
if sx == "Male":
    pr = "False"
bt = st.sidebar.button("Predict Thyroid Risk")
st.title("🫁 Thyroid Prediction App")
st.subheader("🔵 Domain: Endocrinology")
st.write("AIM: To predict Thyroid using Machine Learning and track patient data.")
st.write("UI: Using Streamlit - all UI components (side bar, meter bar, graph rendering.)")
st.write("Backend: Ensmebled Method (As per  project details given) (4 Models mentioned)")
st.write("Reads the CSV file uploaded, trains 4 distinct Machine Learning Models:")
st.write("• Decision Tree • Random Forest • Support Vector Machine (SVM) • XGBoost")
st.write("Use machine learning to estimate a patient's risk of thyroid abnormalities.")
st.write("⭐ Hosted in Streamlit")
st.write("⭐ Date using Matplotlib's Date Time")
st.write("⭐ Input Dataset Fields")
st.write("•	Age (in years) &nbsp; • Sex/Gender &nbsp; • TSH — Thyroid-Stimulating Hormone &nbsp; • T3 — Triiodo Thyronine Measurement &nbsp; • TT4 — Total Thyroxine &nbsp; • T4U — Thyroxine Uptake Measurement &nbsp; • FTI — Free Thyroxine Index &nbsp; • Pregnant — Pregnancy Indicator")
st.write(" ")
st.subheader("🔵 Technology Used")
st.write("⭐ Algorithms: Decision Tree, Random Forest, SVM (Support Vector Machine), XGBoost (Extra Gradient Boost)")
st.write("⭐ Database: SQLite")
st.write("Uses predictions.db to store patient records from SQLite database in where the data is stored each time as data fetched.")
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
    pred = mdl.predict(row)[0]
    label = le.inverse_transform([pred])[0]
    conf = mdl.predict_proba(row)[0].max() * 100
    save_record((datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S"), ag, sx, ts, t3, tt, tu, ft, ot, tsu, pr, str(label), float(conf)))
    probs = mdl.predict_proba(row)[0]
    classes = list(le.classes_)
    p_ab = 100 - probs[classes.index("normal")] * 100
    ab_idx = max((i for i, c in enumerate(classes) if c != "normal"), key=lambda i: probs[i])
    ab_label = classes[ab_idx]
    if p_ab < 20:
        st.success(f"Normal: no thyroid abnormality detected. Abnormal probability: {p_ab:.1f}%")
    elif p_ab < 50:
        st.warning(f"Low Risk: possible {ab_label}. Abnormal probability: {p_ab:.1f}%")
    else:
        st.error(f"High Risk: likely {ab_label}. Abnormal probability: {p_ab:.1f}%")
    
    st.markdown("### 🔍 Feature Range Indicators")
    ref_ranges = ref
    val_dict = {"TSH": ts, "T3": t3, "TT4": tt, "T4U": tu, "FTI": ft}
    cols_ind = st.columns(len(val_dict))
    for i, (k, v) in enumerate(val_dict.items()):
        low, high = ref_ranges[k]
        if v < low:
            cols_ind[i].metric(k, round(v, 2), "-Low", delta_color="off")
        elif v > high:
            cols_ind[i].metric(k, round(v, 2), "+High", delta_color="inverse")
        else:
            cols_ind[i].metric(k, round(v, 2), "Normal", delta_color="off")
    st.dataframe(pd.DataFrame({"Class": le.classes_, "Probability (%)": mdl.predict_proba(row)[0] * 100}), use_container_width=True)
with st.expander("Dataset check"):
    st.write(df[COL["target"]].value_counts())
    st.write(df[NUM].describe())
    st.write((df["TT4"] / df["T4U"] - df["FTI"]).abs().describe())
    st.write(df.groupby(COL["target"])[["TSH", "T3", "TT4", "T4U", "FTI"]].mean())
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
for mc, n in zip(st.columns(len(acc)), acc):
    mc.metric(n, f"Acc {acc[n]:.1f}%", f"Macro F1 {f1[n]:.1f}%", delta_color="off")
c1, c2 = st.columns(2)
fig1, ax1 = plt.subplots()
ax1.bar(list(acc.keys()), list(acc.values()), color=["#92c5de", "#e8867c", "#a6d96a", "#fdae61", "#8073ac"])
plt.setp(ax1.get_xticklabels(), rotation=20, ha="right")
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
st.subheader("🗃️ Patient Records Over Time")
hist = load_records()
if hist.empty:
    st.info("No records yet. Click Predict Thyroid Risk to save the first record.")
else:
    hist["created_at"] = pd.to_datetime(hist["created_at"])
    param = st.selectbox("Select parameter to visualize", ["tsh", "t3", "tt4", "t4u", "fti", "age", "confidence"])
    plot_df = hist.tail(15)
    col1, col2 = st.columns(2)
    with col1:
        fig_sc, ax_sc = plt.subplots(figsize=(6, 4))
        ax_sc.scatter(plot_df["created_at"], plot_df[param], color="blue")
        ax_sc.set_xlabel("Time")
        ax_sc.set_ylabel(param.upper())
        ax_sc.set_title(f"{param.upper()} Scatter Plot")
        plt.setp(ax_sc.get_xticklabels(), rotation=45, ha="right")
        fig_sc.tight_layout()
        st.pyplot(fig_sc)

    with col2:
        fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
        ax_bar.bar(range(len(plot_df)), plot_df[param], color="orange")
        ax_bar.set_xlabel("Records")
        ax_bar.set_ylabel(param.upper())
        ax_bar.set_title(f"{param.upper()} Bar Graph")
        fig_bar.tight_layout()
        st.pyplot(fig_bar)
    st.dataframe(hist.sort_values("created_at", ascending=False), use_container_width=True, column_config={"created_at": st.column_config.DatetimeColumn("Date & Time", format="DD-MM-YYYY hh:mm:ss a")})
    confirm = st.checkbox("Confirm clear all records")
    if st.button("Clear Database", disabled=not confirm):
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM predictions")
            conn.execute("DELETE FROM sqlite_sequence WHERE name='predictions'")
        st.rerun()
st.divider()
st.write(" ")
st.subheader("💻 Developed by Harish.R")
