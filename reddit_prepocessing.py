import json
import os
import re
from datetime import datetime

try:
    import emoji
except ImportError:
    emoji = None

# ===== 通用清洗逻辑（共同部分）=====
class BasePreprocessor:
    def __init__(self):
        pass

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        text = text.lower().strip()
        text = re.sub(r"http\S+|www\.\S+", "", text)
        text = re.sub(r"<.*?>", "", text)
        text = re.sub(r"\[deleted\]|\[removed\]", "", text)
        if emoji:
            text = emoji.demojize(text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def clean_comment_tree(self, comments):
        cleaned = []
        for idx, comment in enumerate(comments):
            if isinstance(comment, str):
                cleaned_body = self.clean_text(comment)
                if not cleaned_body:
                    continue
                cleaned.append({
                    "comment_id": f"auto_comment_{idx}",
                    "author": "Anonymous",
                    "body": cleaned_body,
                    "score": 0,
                    "created_utc": 0,
                    "replies": []
                })
            elif isinstance(comment, dict) and "comment_id" in comment:
                cleaned_body = self.clean_text(comment.get("body", ""))
                if not cleaned_body:
                    continue
                replies = comment.get("replies", [])
                if isinstance(replies, str):
                    try:
                        replies = json.loads(replies)
                    except json.JSONDecodeError:
                        replies = []
                cleaned.append({
                    "comment_id": comment.get("comment_id"),
                    "author": comment.get("author", "Anonymous"),
                    "body": cleaned_body,
                    "score": comment.get("score", 0),
                    "created_utc": comment.get("created_utc", 0),
                    "replies": self.clean_comment_tree(replies)
                })
        return cleaned

# ===== 专用于主题建模的预处理器 =====
class TopicModelPreprocessor(BasePreprocessor):
    def preprocess(self, data):
        results = []
        seen_ids = set()

        for post in data:
            post_id = post.get("id", "")
            if post_id in seen_ids:
                continue
            seen_ids.add(post_id)

            raw_title = post.get("title", "")
            raw_body = post.get("selftext", "")
            combined_text = self.clean_text(raw_title + " " + raw_body)
            body_text = self.clean_text(raw_body)

            comments = post.get("comments", [])
            if isinstance(comments, str):
                try:
                    comments = json.loads(comments)
                except json.JSONDecodeError:
                    comments = []

            has_valid_comments = any(isinstance(c, dict) and "comment_id" in c for c in comments) or any(isinstance(c, str) for c in comments)
            is_title_only = len(body_text.strip()) == 0
            too_short = len(combined_text.split()) < 10 or len(combined_text) < 15

            if is_title_only and not has_valid_comments:
                continue
            if too_short and not has_valid_comments:
                continue

            post_data = {
                "id": post_id,
                "type": "post",
                "title": raw_title,
                "body": raw_body,
                "content": combined_text,
                "author": post.get("author", "Anonymous"),
                "score": post.get("score", 0),
                "created_utc": post.get("created_utc", 0),
                "is_controversial": post.get("is_controversial", False),
                "comments": self.clean_comment_tree(comments)
            }

            results.append(post_data)
        return results

# ===== 专用于情感分析的预处理器 =====
class SentimentAnalysisPreprocessor(BasePreprocessor):
    def preprocess(self, data):
        results = []
        controversial_comments = []
        seen_ids = set()

        for post in data:
            post_id = post.get("id", "")
            if post_id in seen_ids:
                continue
            seen_ids.add(post_id)

            raw_title = post.get("title", "")
            raw_body = post.get("selftext", "")
            combined_text = self.clean_text(raw_title + " " + raw_body)
            is_controversial = post.get("is_controversial", False)

            comments = post.get("comments", [])
            if isinstance(comments, str):
                try:
                    comments = json.loads(comments)
                except json.JSONDecodeError:
                    comments = []

            cleaned_comments = self.clean_comment_tree(comments)

            post_data = {
                "id": post_id,
                "type": "post",
                "title": raw_title,
                "body": raw_body,
                "content": combined_text,
                "author": post.get("author", "Anonymous"),
                "score": post.get("score", 0),
                "created_utc": post.get("created_utc", 0),
                "is_controversial": is_controversial,
                "comments": cleaned_comments
            }

            results.append(post_data)

            if is_controversial:
                controversial_comments.extend(cleaned_comments)

        return results, controversial_comments

# ===== 执行预处理和保存文件 =====
if __name__ == "__main__":
    input_path = "data-collection/reddit_threebody_filtered1_marked.json"
    output_dir = "reddit_cleaned_data/"
    os.makedirs(output_dir, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    topic_processor = TopicModelPreprocessor()
    sentiment_processor = SentimentAnalysisPreprocessor()

    topic_data = topic_processor.preprocess(raw_data)
    sentiment_data, _ = sentiment_processor.preprocess(raw_data)

    with open(os.path.join(output_dir, "reddit_for_topic_modeling.json"), "w", encoding="utf-8") as f:
        json.dump(topic_data, f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_dir, "reddit_for_sentiment_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(sentiment_data, f, ensure_ascii=False, indent=2)

    print(f" Preprocessing complete. Files saved to {output_dir}")
    print(f"Topic modeling data: {len(topic_data)} entries")
    print(f"Sentiment analysis data: {len(sentiment_data)} entries")
