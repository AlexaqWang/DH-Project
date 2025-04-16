import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from keybert import KeyBERT
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ========== 0. Set GPT API KEY ==========
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Font configuration
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ========== 1. Load Data ==========
file_path = "reddit_cleaned_data/reddit_for_topic_modeling.json"
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)
df.rename(columns={"created_utc": "timestamp", "score": "upvotes"}, inplace=True)
df["upvotes"] = df["upvotes"].fillna(0).astype(int)
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

# ========== 3. Compute Weights ==========
df["upvote_weight"] = df["upvotes"] / df["upvotes"].max()
df["weight"] = df["upvote_weight"].fillna(0.05).clip(lower=0.05)

# ========== 4. Embeddings ==========
print("Computing sentence embeddings...")
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = np.array([
    model.encode(text, show_progress_bar=False)
    for text in tqdm(df["content"].tolist(), desc="Generating embeddings")
])
weights = df["weight"].values[:, np.newaxis]
weighted_embeddings = embeddings * weights
df = df.reset_index(drop=True)

# ========== 5. Topic Modeling ==========
print("Training BERTopic model...")
vectorizer_model = CountVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=1,
    max_df=0.9
)

topic_model = BERTopic(
    embedding_model=model,
    vectorizer_model=vectorizer_model,
    representation_model=KeyBERTInspired(),
    calculate_probabilities=False,
    verbose=True,
    min_topic_size=10,
    nr_topics="auto"
)
topics, _ = topic_model.fit_transform(df["content"].tolist(), embeddings=weighted_embeddings)
df["topic"] = topics

# ========== 6. Keyword Extraction ==========
print("Extracting keywords with KeyBERT...")
kw_model = KeyBERT(model=model)
df["keywords"] = df["content"].apply(
    lambda x: [kw[0] for kw in kw_model.extract_keywords(x, top_n=5, stop_words="english")]
)

# ========== 7. Topic Divergence ==========
print("Computing topic divergence (within-topic embedding disagreement)...")
def compute_topic_divergence(topic_id, df, embeddings):
    idx = df[df["topic"] == topic_id].index
    if len(idx) < 2:
        return 0
    topic_embeddings = embeddings[idx]
    sim_matrix = cosine_similarity(topic_embeddings)
    avg_similarity = (np.sum(sim_matrix) - len(sim_matrix)) / (len(sim_matrix)**2 - len(sim_matrix))
    return 1 - avg_similarity

topic_divergence_scores = {
    topic: compute_topic_divergence(topic, df, embeddings)
    for topic in df["topic"].unique()
}
df["divergence_score"] = df["topic"].map(topic_divergence_scores)

# ========== 8. Visualization ==========
topic_counts = df["topic"].value_counts().sort_index()
topic_likes = df.groupby("topic")["upvotes"].mean().sort_values(ascending=False)
topic_df = pd.DataFrame({
    "topic": topic_divergence_scores.keys(),
    "divergence_score": topic_divergence_scores.values(),
    "frequency": topic_counts.values,
})
topic_df["share"] = topic_df["frequency"] / topic_df["frequency"].sum()

# Plot 1: Topic Distribution
plt.figure(figsize=(12, 6))
plt.bar(topic_counts.index, topic_counts.values, color="skyblue")
plt.xlabel("Topic ID")
plt.ylabel("Number of Posts")
plt.title("Topic Distribution")
plt.tight_layout()
plt.show()

# Plot 2: Popularity
plt.figure(figsize=(14, 6))
sns.barplot(x=topic_likes.index, y=topic_likes.values, palette="viridis")
plt.xlabel("Topic ID")
plt.ylabel("Average Upvotes")
plt.title("Most Popular Topics (by Avg Upvotes)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot 3: Divergence
top_n = 20
top_div = topic_df.sort_values("divergence_score", ascending=False).head(top_n)
plt.figure(figsize=(12, 6))
sns.barplot(x="topic", y="divergence_score", data=top_div, palette="coolwarm")
plt.title(f"Top {top_n} Most Controversial Topics (by Divergence Score)")
plt.xlabel("Topic ID")
plt.ylabel("Divergence Score")
plt.tight_layout()
plt.show()

# Plot 4: Share vs Divergence
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=topic_df,
    x="share", y="divergence_score",
    size="frequency", hue="divergence_score",
    palette="coolwarm", sizes=(20, 200),
    alpha=0.7, edgecolor="gray"
)
plt.title("Topic Divergence vs Share of Corpus")
plt.xlabel("Share of All Posts (Topic Frequency)")
plt.ylabel("Divergence Score")
plt.legend(title="Divergence", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

# ========== 9. GPT Summary + Type ==========
print("Generating summary table of top 10 most upvoted topics...")
rep_docs = topic_model.get_representative_docs()
top_10_topic_ids = topic_likes.head(10).index.tolist()

def gpt_topic_summary_and_type(topic_id, docs):
    joined = "\n".join(f"{i+1}. {doc}" for i, doc in enumerate(docs[:3]))
    prompt = f"""
You are analyzing Reddit discussions about the novel The Three-Body Problem.
Summarize the topic represented by the following comments in one sentence, and classify it as one of:
[Character, Plot, Concept, Opinion].

Reddit comments:
{joined}

Return in this format:
Summary: ...
Type: ...
""".strip()

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for text analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=150
        )
        content = response.choices[0].message.content
        lines = content.strip().split("\n")
        summary = next((l for l in lines if l.lower().startswith("summary:")), "").replace("Summary:", "").strip()
        topic_type = next((l for l in lines if l.lower().startswith("type:")), "").replace("Type:", "").strip()
        return summary, topic_type
    except Exception as e:
        print(f"GPT error for topic {topic_id}: {e}")
        return "N/A", "Unknown"

top_topic_data = []
for topic_id in top_10_topic_ids:
    docs = rep_docs.get(topic_id, [])
    summary, topic_type = gpt_topic_summary_and_type(topic_id, docs)
    top_topic_data.append({
        "Topic ID": topic_id,
        "Average Upvotes": round(topic_likes[topic_id], 2),
        "Divergence Score": round(topic_divergence_scores.get(topic_id, 0), 4),
        "Topic Summary": summary,
        "Topic Type": topic_type
    })

top_topics_df = pd.DataFrame(top_topic_data)
print("\nTop 10 Topics by Average Upvotes:")
print(top_topics_df)

# ========== 10. Save Results ==========
output_dir = "reddit_analysis_results"
os.makedirs(output_dir, exist_ok=True)
top_topics_df.to_csv(os.path.join(output_dir, "top_topics_table_with_gpt.csv"), index=False)
df.to_csv(os.path.join(output_dir, "reddit_with_topics_keywords_divergence.csv"), index=False)

print("All analysis and GPT summaries saved to:", output_dir)
