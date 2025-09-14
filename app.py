import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Title
st.title("Early Student Risk Alert System (with ML)")

# Upload CSVs
attendance_file = st.file_uploader("Upload Attendance CSV", type=["csv"])
scores_file = st.file_uploader("Upload Scores CSV", type=["csv"])
fees_file = st.file_uploader("Upload Fees CSV", type=["csv"])

if attendance_file and scores_file and fees_file:
    # Load data
    attendance = pd.read_csv(attendance_file)
    scores = pd.read_csv(scores_file)
    fees = pd.read_csv(fees_file)

    # Merge
    merged = attendance.merge(scores, on="student_id").merge(fees, on="student_id")

    # --- Rule-based risk (existing logic) ---
    def get_risk(row):
        if row["attendance_pct"] < 60 or row["avg_score"] < 40 or row["fees_status"] == "Unpaid":
            return "High"
        elif row["attendance_pct"] < 75 or row["avg_score"] < 50:
            return "Medium"
        else:
            return "Low"

    merged["rule_risk"] = merged.apply(get_risk, axis=1)

    # --- Machine Learning (new) ---
    # Features
    X = merged[["attendance_pct", "avg_score"]].copy()
    X["fees_unpaid"] = merged["fees_status"].apply(lambda x: 1 if x == "Unpaid" else 0)

    # Labels from rule-based risk
    y = merged["rule_risk"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train Random Forest
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Predictions
    merged["ml_predicted_risk"] = clf.predict(X)

       # Show table with colored risk levels
    st.subheader("Merged Student Data with Risk Levels")

    def color_risk(val):
        if val == "Low":
            return "background-color: green; color: black"
        elif val == "Medium":
            return "background-color: orange; color: black"
        elif val == "High":
            return "background-color: red; color: white"
        return ""

    styled_df = merged[[
        "student_id","name","attendance_pct","avg_score","fees_status","rule_risk","ml_predicted_risk"
    ]].style.applymap(color_risk, subset=["rule_risk","ml_predicted_risk"])

    st.dataframe(styled_df, use_container_width=True)


    styled_df = merged[[
        "student_id","name","attendance_pct","avg_score","fees_status","rule_risk","ml_predicted_risk"
    ]].style.applymap(color_risk, subset=["rule_risk","ml_predicted_risk"])

    st.dataframe(styled_df, use_container_width=True)

    # Chart: ML Risk distribution
    st.subheader("ML Risk Level Distribution")
    risk_counts = merged["ml_predicted_risk"].value_counts()
    fig, ax = plt.subplots()
    risk_counts.plot(kind="bar", ax=ax)
    ax.set_xlabel("Risk Level")
    ax.set_ylabel("Number of Students")
    st.pyplot(fig)

    # Show ML performance vs rules
    st.subheader("Model Evaluation (ML vs Rule-based labels)")
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=False)
    st.text(report)

import requests

# Hugging Face summarization
API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
headers = {"Authorization": f"Bearer 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"}

def summarize_text(text):
    response = requests.post(API_URL, headers=headers, json={"inputs": text})
    if response.status_code == 200:
        return response.json()[0]["summary_text"]
    else:
        return f"Error: {response.text}"

# Generate summary of ML table
st.subheader("AI Summary of Risk Table")
table_text = merged.to_string(index=False)
summary = summarize_text(table_text[:1500])  # limit to avoid long payloads
st.write(summary)


