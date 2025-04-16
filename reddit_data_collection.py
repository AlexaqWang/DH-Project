import praw
import json
import time
from tqdm import tqdm
from prawcore.exceptions import RequestException, ResponseException, TooManyRequests
from datetime import datetime

load_dotenv()  

REDDIT_API = {
    "client_id": os.getenv("REDDIT_CLIENT_ID"),
    "client_secret": os.getenv("REDDIT_CLIENT_SECRET"),
    "user_agent": os.getenv("REDDIT_USER_AGENT")
}
SAVE_PATH = "data-collection/reddit_threebody_filtered1.json"
EXCLUDED_FLAIRS = {"Discussion - TV Series", "Meme", "News"}


class APIScraper:
    def __init__(self):
        self.posts_data = []
        self.seen_ids = set()
        self.reddit = praw.Reddit(**REDDIT_API)
        self.total_fetched = 0
        self.start_time = datetime.now()
        self.progress_bar = tqdm(desc="Total Posts Fetched", unit=" posts", position=0, leave=True)

    def is_relevant_post(self, submission):
        flair = getattr(submission, "link_flair_text", None)
        if flair in EXCLUDED_FLAIRS:
            return False
        return submission.selftext is not None

    def is_controversial(self, submission):
        try:
            return submission.score > 50 and submission.upvote_ratio < 0.6
        except Exception:
            return False

    def fetch_comments(self, submission):
        try:
            submission.comments.replace_more(limit=None)
            time.sleep(2)
            comment_list = submission.comments.list()
            return [comment.body for comment in comment_list if hasattr(comment, "body")]
        except Exception:
            return []

    def fetch_posts(self):
        try:
            subreddit = self.reddit.subreddit("threebodyproblem")
            sort_methods = ["new", "hot", "top", "rising"]
            last_update_time = time.time()
            max_posts_per_sort = 5000

            for sort_method in sort_methods:
                after = None
                while True:
                    try:
                        if sort_method == "top":
                            submissions = subreddit.top(limit=max_posts_per_sort, time_filter="all", params={"after": after})
                        elif sort_method == "new":
                            submissions = subreddit.new(limit=max_posts_per_sort, params={"after": after})
                        elif sort_method == "hot":
                            submissions = subreddit.hot(limit=max_posts_per_sort, params={"after": after})
                        elif sort_method == "rising":
                            submissions = subreddit.rising(limit=max_posts_per_sort)
                        else:
                            submissions = []

                        for submission in submissions:
                            if submission.id in self.seen_ids:
                                continue
                            self.seen_ids.add(submission.id)

                            if not submission.title or not self.is_relevant_post(submission):
                                continue

                            time.sleep(1)

                            comments = self.fetch_comments(submission) if submission.num_comments > 0 else []

                            post_data = {
                                "id": submission.id,
                                "title": submission.title,
                                "selftext": submission.selftext,
                                "author": getattr(submission.author, "name", "Anonymous"),
                                "flair": getattr(submission, "link_flair_text", "Unknown"),
                                "score": submission.score,
                                "upvote_ratio": getattr(submission, "upvote_ratio", 1.0),
                                "created_utc": submission.created_utc,
                                "num_comments": submission.num_comments,
                                "is_controversial": self.is_controversial(submission),
                                "comments": comments
                            }

                            self.posts_data.append(post_data)
                            self.total_fetched += 1
                            elapsed_time = datetime.now() - self.start_time
                            self.progress_bar.set_description(f"Fetched: {self.total_fetched} | Runtime: {elapsed_time}")
                            self.progress_bar.update(1)

                        if time.time() - last_update_time > 10:
                            elapsed_time = datetime.now() - self.start_time
                            print(f"\n[INFO] Runtime: {elapsed_time} | Total Posts: {self.total_fetched}")
                            last_update_time = time.time()

                        if hasattr(submissions, "after") and submissions.after:
                            after = submissions.after
                            print(f"[DEBUG] Current 'after' value: {after}")
                        else:
                            print(f"[DEBUG] No more pages left in {sort_method}")
                            break

                    except TooManyRequests:
                        print("[WARN] Hit API rate limit, sleeping for 10 seconds...")
                        time.sleep(10)
                    except (RequestException, ResponseException) as e:
                        print(f"[ERROR] API error: {e}, sleeping for 5 seconds...")
                        time.sleep(5)

            print(f"\n[INFO] Total fetched posts: {self.total_fetched}")

        except Exception as e:
            print(f"[ERROR] Critical error: {e}")

    def fetch_controversial_posts_dedup(self):
        """Fetch controversial posts from Reddit's own ranking and add those not already included"""
        subreddit = self.reddit.subreddit("threebodyproblem")
        print("[INFO] Fetching additional controversial posts...")

        try:
            submissions = subreddit.controversial(limit=1000, time_filter="all")
            added_count = 0

            for submission in tqdm(submissions, desc="Merging Controversial Posts", unit=" post"):
                if submission.id in self.seen_ids:
                    continue

                if not submission.title or not self.is_relevant_post(submission):
                    continue

                time.sleep(1)

                comments = self.fetch_comments(submission) if submission.num_comments > 0 else []

                post_data = {
                    "id": submission.id,
                    "title": submission.title,
                    "selftext": submission.selftext,
                    "author": getattr(submission.author, "name", "Anonymous"),
                    "flair": getattr(submission, "link_flair_text", "Unknown"),
                    "score": submission.score,
                    "upvote_ratio": getattr(submission, "upvote_ratio", 1.0),
                    "created_utc": submission.created_utc,
                    "num_comments": submission.num_comments,
                    "is_controversial": self.is_controversial(submission),
                    "comments": comments
                }

                self.posts_data.append(post_data)
                self.seen_ids.add(submission.id)
                added_count += 1

            print(f"[INFO] Added {added_count} new controversial posts from Reddit ranking")

        except Exception as e:
            print(f"[ERROR] Failed to fetch controversial posts: {e}")

    def save_data(self):
        try:
            json_data = json.dumps(self.posts_data, ensure_ascii=False, indent=2)
            with open(SAVE_PATH, "w", encoding="utf-8") as f:
                f.write(json_data)
            print(f"\n[INFO] Data saved to {SAVE_PATH}")
        except Exception as e:
            print(f"[ERROR] Error saving data: {e}")

    def print_runtime(self):
        end_time = datetime.now()
        elapsed_time = end_time - self.start_time
        print(f"\n[INFO] Total runtime: {elapsed_time}")


if __name__ == "__main__":
    scraper = APIScraper()
    scraper.fetch_posts()
    scraper.fetch_controversial_posts_dedup()
    scraper.save_data()
    scraper.print_runtime()
