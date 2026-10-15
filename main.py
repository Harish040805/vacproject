# Without UI, deploy in Github and call in Render/Vercel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import streamlit as st

st.title("🫁 Thyroid Prediction App")
st.write("🌐 Welcome to the application! Thyroid Prediction App developed using Python and Deployed using GitHub and Streamlit!")
st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")
st.write("💻 Developed by Harish.R")

def run_thyroid_pipeline(df):
    X = df.drop(columns=['thyroid_class'])
    y = df['thyroid_class']
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    X = pd.get_dummies(X, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    models = {
        "🌳 DECISION TREE": DecisionTreeClassifier(random_state=42),
        "🌲 RANDOM FOREST": RandomForestClassifier(n_estimators=100, random_state=42),
        "⚙️ SVM": SVC(kernel='rbf', probability=True, random_state=42),
        "⚡️ XGBOOST": XGBClassifier(eval_metric='mlogloss', random_state=42)
    }
    results = {}
    for name, model in models.items():
        if "SVM" in name:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        print("="*60)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        print(" ")
        print(f"{name}")
        print(" ")
        print(f"🔴 Accuracy: {acc:.4f}")
        print(" ")
        report = classification_report(y_test, y_pred, zero_division=0)
        report = report.replace("precision", "Precision")
        report = report.replace("recall", "Recall")
        report = report.replace("f1-score", "F1-Score")
        report = report.replace("support", "Support")
        report = report.replace("accuracy", "Accuracy")
        report = report.replace("macro avg", "Macro Avg")
        report = report.replace("weighted avg", "Weighted Avg")
        print(report)
    return results
if __name__ == "__main__":
    df = pd.read_csv("thyroid_dataset.csv")
    results = run_thyroid_pipeline(df)
    print("="*60)
