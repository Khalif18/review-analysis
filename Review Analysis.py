#!/usr/bin/env python3

import os
import json
import math
import re
import time
import pandas as pd
from pathlib import Path
from datetime import datetime
from together import Together

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

INPUT_FILE = Path(r"C:\Users\mdkha\OneDrive\Desktop\review analysis\Google_Reviews_Analysis_POC (3).xlsx")
OUTPUT_DIR = Path(r"C:\Users\mdkha\OneDrive\Desktop\review analysis\Output")

MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
BATCH_SIZE = 10
MAX_RETRIES = 3
RETRY_DELAY = 2

# --------------------------------------------------
# SETUP
# --------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FILE = OUTPUT_DIR / f"Google_Review_AI_Analysis_{timestamp}.xlsx"

api_key = os.getenv("TOGETHER_API_KEY", "605aa6c5baf53f6dab760f2dcc022ada792883aceaeeb90d43071573e1e6d2b9")
client = Together(api_key=api_key)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_excel(INPUT_FILE)
df["Date"] = pd.to_datetime(df["Date"])
df["Month"] = df["Date"].dt.to_period("M")

total_records = len(df)
total_batches = math.ceil(total_records / BATCH_SIZE)

print(f"\nTotal Reviews: {total_records}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Total Batches: {total_batches}\n")

# --------------------------------------------------
# NORMALIZATION FUNCTION
# --------------------------------------------------

def normalize_issue(issue_text):
    if not issue_text or issue_text in ["None", "Parsing Error", "Unclassified"]:
        return "None"

    text = issue_text.lower()

    if "air" in text and "condition" in text:
        return "Air Conditioning Failure"
    if "escalator" in text:
        return "Escalator Failure"
    if "lift" in text or "elevator" in text:
        return "Lift Failure"
    if "clean" in text or "hygiene" in text or "odor" in text:
        return "Cleanliness Issue"
    if "parking" in text:
        return "Parking Management Issue"
    if "security" in text:
        return "Security Concern"
    if "maintenance" in text or "slow" in text:
        return "Maintenance Response Delay"

    return "Other Operational Issue"

# --------------------------------------------------
# SAFE JSON EXTRACTION
# --------------------------------------------------

def extract_json_array(text):
    match = re.search(r"\[.*\]", text, re.DOTALL)
    return match.group() if match else None

# --------------------------------------------------
# BATCH ANALYSIS
# --------------------------------------------------

def analyze_batch(batch_reviews):

    prompt = f"""
Return ONLY valid JSON array.
You MUST return exactly {len(batch_reviews)} objects.
Do NOT add explanation or markdown.

Format:
[
  {{
    "sentiment": "Positive/Neutral/Negative",
    "sentiment_score": 0-1,
    "primary_issue_category": "Short Category Name or None",
    "pain_point": "Short Pain Description or None",
    "confidence_score": 0-1
  }}
]
"""

    for i, review in enumerate(batch_reviews):
        prompt += f"\nReview {i+1}: {review}\n"

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=3000
            )

            raw_output = response.choices[0].message.content.strip()
            json_text = extract_json_array(raw_output)

            if not json_text:
                raise ValueError("No JSON found")

            results = json.loads(json_text)

            if len(results) != len(batch_reviews):
                raise ValueError("Length mismatch")

            return results

        except Exception as e:
            print(f"Retry {attempt+1}/{MAX_RETRIES} failed: {e}")
            time.sleep(RETRY_DELAY)

    print("⚠ Batch failed after retries — applying fallback.\n")

    return [{
        "sentiment": "Unknown",
        "sentiment_score": 0.5,
        "primary_issue_category": "Unclassified",
        "pain_point": "Parsing Error",
        "confidence_score": 0.5
    }] * len(batch_reviews)

# --------------------------------------------------
# PROCESS REVIEWS
# --------------------------------------------------

analysis_results = []

for batch_num in range(total_batches):

    start_idx = batch_num * BATCH_SIZE
    end_idx = min(start_idx + BATCH_SIZE, total_records)

    batch_df = df.iloc[start_idx:end_idx]
    batch_reviews = batch_df["Review"].tolist()

    print(f"Processing Batch {batch_num+1}/{total_batches} "
          f"({end_idx}/{total_records})")

    batch_results = analyze_batch(batch_reviews)

    for i in range(len(batch_df)):
        row = batch_df.iloc[i]
        ai_result = batch_results[i]

        normalized_pain = normalize_issue(
            ai_result.get("pain_point", "None")
        )

        analysis_results.append({
            "MallName": row["MallName"],
            "Date": row["Date"],
            "Month": row["Month"],
            "Review": row["Review"],
            "Rating": row["Rating"],
            "Sentiment": ai_result.get("sentiment", "Unknown"),
            "SentimentScore": ai_result.get("sentiment_score", 0.5),
            "PainPoint": normalized_pain,
            "ConfidenceScore": ai_result.get("confidence_score", 0.5)
        })

    progress = round((end_idx / total_records) * 100, 2)
    print(f"Completed {progress}%\n")

analysis_df = pd.DataFrame(analysis_results)

# --------------------------------------------------
# FULL ISSUE TRACKER
# --------------------------------------------------

issue_tracker = []

grouped = analysis_df[analysis_df["PainPoint"] != "None"].groupby(
    ["MallName", "PainPoint"]
)

for (mall, issue), group in grouped:

    first_seen = group["Date"].min()
    last_seen = group["Date"].max()
    frequency = len(group)

    recent_30 = group[
        group["Date"] >= (datetime.now() - pd.Timedelta(days=30))
    ]
    recent_count = len(recent_30)

    age_days = (datetime.now() - first_seen).days

    if recent_count == 0:
        status = "Resolved"
    elif recent_count < (frequency / 3):
        status = "Improving"
    else:
        status = "Open / Escalating"

    evidence_reviews = (
        group.sort_values("Date", ascending=False)
        .head(3)["Review"]
        .tolist()
    )

    avg_confidence = round(group["ConfidenceScore"].mean(), 2)

    issue_tracker.append({
        "MallName": mall,
        "PainPoint": issue,
        "FirstSeen": first_seen,
        "LastSeen": last_seen,
        "TotalOccurrences": frequency,
        "Last30DaysCount": recent_count,
        "AgeDays": age_days,
        "Status": status,
        "AvgConfidenceScore": avg_confidence,
        "Evidence1": evidence_reviews[0] if len(evidence_reviews) > 0 else "",
        "Evidence2": evidence_reviews[1] if len(evidence_reviews) > 1 else "",
        "Evidence3": evidence_reviews[2] if len(evidence_reviews) > 2 else ""
    })

issue_tracker_df = pd.DataFrame(issue_tracker)

# --------------------------------------------------
# BOARD LEVEL SUMMARY
# --------------------------------------------------

board_summary = []

for mall in analysis_df["MallName"].unique():

    mall_data = analysis_df[analysis_df["MallName"] == mall]

    total = len(mall_data)
    positive = len(mall_data[mall_data["Sentiment"] == "Positive"])
    negative = len(mall_data[mall_data["Sentiment"] == "Negative"])

    positive_pct = round((positive / total) * 100, 2)

    top_issues = (
        mall_data[mall_data["PainPoint"] != "None"]
        .groupby("PainPoint")
        .size()
        .sort_values(ascending=False)
        .head(2)
    )

    top_1 = top_issues.index[0] if len(top_issues) > 0 else "No Major Issue"
    top_2 = top_issues.index[1] if len(top_issues) > 1 else "None"

    monthly_sentiment = (
        mall_data.groupby("Month")["SentimentScore"].mean().sort_index()
    )

    if len(monthly_sentiment) >= 2:
        if monthly_sentiment.iloc[-1] > monthly_sentiment.iloc[-2]:
            trajectory = "Improving"
        elif monthly_sentiment.iloc[-1] < monthly_sentiment.iloc[-2]:
            trajectory = "Declining"
        else:
            trajectory = "Stable"
    else:
        trajectory = "Stable"

    escalation_flag = "Yes" if (len(top_issues) > 0 and top_issues.iloc[0] > 10) else "No"

    if positive_pct >= 70:
        risk = "Low"
    elif positive_pct >= 50:
        risk = "Medium"
    else:
        risk = "High"

    narrative = (
        f"{mall} shows {positive_pct}% positive sentiment. "
        f"Top issues: {top_1} and {top_2}. "
        f"Sentiment trend is {trajectory}. "
        f"Risk classified as {risk}. "
        f"Escalation required: {escalation_flag}."
    )

    board_summary.append({
        "MallName": mall,
        "PositivePercentage": positive_pct,
        "TopIssue1": top_1,
        "TopIssue2": top_2,
        "SentimentTrajectory": trajectory,
        "RiskClassification": risk,
        "EscalationFlag": escalation_flag,
        "ExecutiveNarrative": narrative
    })

board_summary_df = pd.DataFrame(board_summary)

# --------------------------------------------------
# EXPORT TO EXCEL
# --------------------------------------------------

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    analysis_df.to_excel(writer, sheet_name="Review_Level_Analysis", index=False)
    issue_tracker_df.to_excel(writer, sheet_name="Issue_Tracker", index=False)
    board_summary_df.to_excel(writer, sheet_name="Board_Level_Summary", index=False)

print(f"\n✅ AI Executive Report Generated Successfully:\n{OUTPUT_FILE}")
