import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import requests

# Title
st.title("Early Student Risk Alert System (with ML + AI Summary)")

# Hugging Face API (optional)
API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
HF_TOKEN = st.secrets.get("HF_TOKEN", None)  # add in Streamlit secrets
headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

def summarize_text(text):
    """Use Hugging Face API if token exists, otherwise fallback."""
    if not HF_TOKEN:
        return None
    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": text})
        if response.status_code == 200:
            return response.json()[0]["summary_text"]
    except Exception as e:
        return None
    return None

# Upload CSVs
attendance_file = st.file_uploader("Upload Attendance CSV", type=["csv"])
scores_file = st.file_uploader("Upload Scores CSV", type=["csv"])
fees_file = st.file_uploader("Upload Fees CSV", type=["csv"])

if attendance_file and scores_file and fees_file:
    # Load uploaded data
    attendance = pd.read_csv(attendance_file)
    scores = pd.read_csv(scores_file)
    fees = pd.read_csv(fees_file)
else:
    st.info("No files uploaded — using sample demo data.")
    attendance = pd.DataFrame([
        {"student_id":101,"name":"Rahul","attendance_pct":85},
        {"student_id":102,"name":"Arun","attendance_pct":55},
        {"student_id":103,"name":"Drake","attendance_pct":70},
        {"student_id":104,"name":"Drisha","attendance_pct":40},
        {"student_id":105,"name":"Kendrick","attendance_pct":95},
    ])
    scores = pd.DataFrame([
        {"student_id":101,"avg_score":78,"prev_score":80,"attempts_failed":0},
        {"student_id":102,"avg_score":35,"prev_score":50,"attempts_failed":1},
        {"student_id":103,"avg_score":60,"prev_score":62,"attempts_failed":0},
        {"student_id":104,"avg_score":30,"prev_score":45,"attempts_failed":2},
        {"student_id":105,"avg_score":90,"prev_score":88,"attempts_failed":0},
    ])
    fees = pd.DataFrame([
        {"student_id":101,"fees_status":"Paid"},
        {"student_id":102,"fees_status":"Unpaid"},
        {"student_id":103,"fees_status":"Paid"},
        {"student_id":104,"fees_status":"Unpaid"},
        {"student_id":105,"fees_status":"Paid"},
    ])


    # Merge
    merged = attendance.merge(scores, on="student_id").merge(fees, on="student_id")

    #Rule-based risk (existing logic)
    def get_risk(row):
        if row["attendance_pct"] < 60 or row["avg_score"] < 40 or row["fees_status"] == "Unpaid":
            return "High"
        elif row["attendance_pct"] < 75 or row["avg_score"] < 50:
            return "Medium"
        else:
            return "Low"

    merged["rule_risk"] = merged.apply(get_risk, axis=1)

    # Machine Learning (new)
    X = merged[["attendance_pct", "avg_score"]].copy()
    X["fees_unpaid"] = merged["fees_status"].apply(lambda x: 1 if x == "Unpaid" else 0)
    y = merged["rule_risk"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    merged["ml_predicted_risk"] = clf.predict(X)

    #Styled Table
    st.subheader("Merged Student Data with Risk Levels")

    def color_risk(val):
        if val == "Low":
            return "background-color: lightgreen; color: black"
        elif val == "Medium":
            return "background-color: orange; color: black"
        elif val == "High":
            return "background-color: red; color: white"
        return ""

    styled_df = merged[[
        "student_id","name","attendance_pct","avg_score","fees_status","rule_risk","ml_predicted_risk"
    ]].style.applymap(color_risk, subset=["rule_risk","ml_predicted_risk"])

    st.dataframe(styled_df, use_container_width=True)

    # Chart
    st.subheader("ML Risk Level Distribution")
    risk_counts = merged["ml_predicted_risk"].value_counts()
    fig, ax = plt.subplots()
    risk_counts.plot(kind="bar", ax=ax, color=["green","orange","red"])
    ax.set_xlabel("Risk Level")
    ax.set_ylabel("Number of Students")
    st.pyplot(fig)

    #  ML Performance
    st.subheader("Model Evaluation (ML vs Rule-based labels)")
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=False)
    st.text(report)

    #  AI Summary 
    st.subheader("AI Summary of Risk Table")

    # Extract key insights
    high_risk_students = merged[merged["ml_predicted_risk"] == "High"]["name"].tolist()
    low_attendance = merged[merged["attendance_pct"] < 60]["name"].tolist()
    unpaid_fees = merged[merged["fees_status"] == "Unpaid"]["name"].tolist()
    low_scores = merged[merged["avg_score"] < 40]["name"].tolist()

    insight_text = (
        f"Out of {len(merged)} students, {len(high_risk_students)} are predicted High Risk: {', '.join(high_risk_students)}. "
        f"Low attendance: {', '.join(low_attendance)}. "
        f"Unpaid fees: {', '.join(unpaid_fees)}. "
        f"Low scores: {', '.join(low_scores)}. "
    )

    summary = summarize_text(insight_text)

    if summary:
        st.write(summary)
    else:
        # Fallback rule-based summary
        summary_text = (
            f"There are {len(merged)} students. "
            f"High risk: {', '.join(high_risk_students) if high_risk_students else 'None'}. "
            f"Low attendance: {', '.join(low_attendance) if low_attendance else 'None'}. "
            f"Unpaid fees: {', '.join(unpaid_fees) if unpaid_fees else 'None'}. "
            f"Low scores: {', '.join(low_scores) if low_scores else 'None'}."
        )
        st.write(summary_text)

