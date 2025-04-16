import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

# ====== 1. Load Cleaned Data ======
input_file = "reddit_cleaned_data/cleaned_sentiment.json"
with open(input_file, "r", encoding="utf-8") as f:
    sentiment_data = json.load(f)

df = pd.DataFrame(sentiment_data)

# ====== 2. Initialize Models ======
vader = SentimentIntensityAnalyzer()
bert = pipeline("sentiment-analysis")

# ====== 3. Sentiment Scoring Functions ======
def get_vader_sentiment(text):
    score = vader.polarity_scores(text)['compound']
    if score > 0.05:
        label = "positive"
    elif score < -0.05:
        label = "negative"
    else:
        label = "neutral"
    return score, label

def apply_bert_with_threshold(texts, threshold=0.7):
    results = bert(texts)
    labels = []
    for r in results:
        if r["score"] < threshold:
            labels.append("neutral")
        else:
            labels.append(r["label"].lower())
    return labels

# ====== 4. Apply VADER and BERT ======
vader_results = df["content"].apply(get_vader_sentiment)
df["vader_score"] = vader_results.apply(lambda x: x[0])
df["vader_sentiment"] = vader_results.apply(lambda x: x[1])

neutral_mask = df["vader_sentiment"] == "neutral"
neutral_texts = df.loc[neutral_mask, "content"].tolist()
df.loc[neutral_mask, "bert_sentiment"] = apply_bert_with_threshold(neutral_texts, threshold=0.7)

# ====== 5. Combine Final Sentiment ======
df["final_sentiment"] = df["vader_sentiment"]
df.loc[neutral_mask, "final_sentiment"] = df.loc[neutral_mask, "bert_sentiment"]

# ====== 6. Save Analysis Result ======
output_dir = "reddit_cleaned_data/"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "final_sentiment_analysis.json")

df[[
    "id", "author", "type", "parent_id", "content",
    "vader_score", "vader_sentiment", "bert_sentiment", "final_sentiment"
]].to_json(output_file, orient="records", indent=4, force_ascii=False)

print(f"Sentiment results saved to {output_file}")

# ====== 7. Prepare Data for Visualization ======
df = df[df["vader_score"].notnull()].copy()
df = df.reset_index(drop=True)
df_filtered = df[~((df["vader_score"] == 0) & (df["final_sentiment"] != "neutral"))].copy()

low_confidence_count = df.loc[neutral_mask, "bert_sentiment"].value_counts().get("neutral", 0)
print(f"{low_confidence_count} texts kept as neutral due to low BERT confidence.")

# ====== 8. Check BERT-Reclassified Neutral Samples ======
removed = df[(df["vader_score"] == 0) & (df["final_sentiment"] != "neutral")]
print(f"\nRemoved {len(removed)} misleading 0-score samples (reclassified by BERT)")
for i, row in removed.head(10).iterrows():
    print(f"\nSample {i}")
    print(f"Final Sentiment: {row['final_sentiment']}")
    print(f"Content: {row['content']}")

# ====== 9. Histogram of VADER Scores (Filtered) ======
plt.figure(figsize=(10, 5))
sns.histplot(data=df_filtered, x="vader_score", bins=40, kde=True, color='skyblue')
plt.axvline(0, color='gray', linestyle='--')
plt.title("Distribution of VADER Scores (Filtered)")
plt.xlabel("VADER Compound Score")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# ====== 10. Final Sentiment Label Distribution ======
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="final_sentiment", order=["positive", "neutral", "negative"], palette="Set2")
plt.title("Final Sentiment Label Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
