import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Title
st.title("Early Student Risk Alert System")

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

    # Risk logic
    def get_risk(row):
        if row["attendance_pct"] < 60 or row["avg_score"] < 40 or row["fees_status"] == "Unpaid":
            return "High"
        elif row["attendance_pct"] < 75 or row["avg_score"] < 50:
            return "Medium"
        else:
            return "Low"

    merged["risk_level"] = merged.apply(get_risk, axis=1)

    # Show table
    st.subheader("Merged Student Data with Risk Levels")
    st.dataframe(merged)

    # Chart: Risk distribution
    st.subheader("Risk Level Distribution")
    risk_counts = merged["risk_level"].value_counts()
    fig, ax = plt.subplots()
    risk_counts.plot(kind="bar", ax=ax)
    ax.set_xlabel("Risk Level")
    ax.set_ylabel("Number of Students")
    st.pyplot(fig)
