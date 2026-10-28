import argparse
import csv
import math
import os
import pathlib
import sys

import numpy as np
import pandas as pd
import praw


TARGET_RECORDS = 25_000
MAX_SUBMISSIONS = 1_000


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Collect Reddit user-post interactions and write JODIE/TGN-compatible "
            "raw and TIGER-ready datasets."
        )
    )
    parser.add_argument("--subreddit", default="MachineLearning")
    parser.add_argument("--target", type=int, default=TARGET_RECORDS)
    parser.add_argument(
        "--max-submissions",
        type=int,
        default=MAX_SUBMISSIONS,
        help="Number of newest subreddit submissions to inspect (maximum 1000).",
    )
    parser.add_argument(
        "--max-more-per-post",
        type=int,
        default=None,
        help="Maximum MoreComments expansions per post; default expands all available.",
    )
    parser.add_argument("--output-dir", default="data")
    return parser


def required_credential(name):
    value = os.environ.get(name)
    if value:
        return value

    try:
        from kaggle_secrets import UserSecretsClient
    except ImportError as exc:
        raise EnvironmentError(
            f"Missing {name}. Set it as an environment variable or add it to "
            "Kaggle Notebook Secrets."
        ) from exc

    value = UserSecretsClient().get_secret(name)
    if not value:
        raise EnvironmentError(
            f"Kaggle Secret {name} is empty. Add it in Notebook options > Secrets."
        )
    return value


def collect_interactions(reddit, subreddit_name, target, max_submissions, max_more):
    user_ids = {}
    item_ids = {}
    seen_comment_ids = set()
    events = []
    subreddit = reddit.subreddit(subreddit_name)

    def get_user_id(author):
        if author is None:
            return None
        username = getattr(author, "name", None)
        if not username or username in {"[deleted]", "[removed]"}:
            return None
        if username not in user_ids:
            user_ids[username] = len(user_ids) + 1
        return user_ids[username]

    for submission in subreddit.new(limit=max_submissions):
        submission_id = submission.id
        if submission_id not in item_ids:
            item_ids[submission_id] = len(item_ids) + 1
        submission_time = float(submission.created_utc)

        submission.comments.replace_more(limit=max_more)
        for comment in submission.comments.list():
            comment_id = getattr(comment, "id", None)
            if not comment_id or comment_id in seen_comment_ids:
                continue
            seen_comment_ids.add(comment_id)

            user_id = get_user_id(getattr(comment, "author", None))
            if user_id is None:
                continue
            try:
                timestamp = float(comment.created_utc)
                score = max(0, int(comment.score))
                depth = max(0, int(comment.depth))
            except (AttributeError, TypeError, ValueError, OverflowError):
                continue
            if not math.isfinite(timestamp) or timestamp < submission_time:
                continue

            # Keep numeric interaction features only; do not persist comment text.
            features = [
                math.log1p(score),
                float(depth),
                math.log1p(timestamp - submission_time),
                float(bool(getattr(comment, "is_submitter", False))),
            ]
            events.append(
                {
                    "user_id": user_id,
                    "item_id": item_ids[submission_id],
                    "timestamp": timestamp,
                    "state_label": 0.0,
                    "features": features,
                }
            )
            if len(events) >= target:
                break
        if len(events) >= target:
            break

    if len(events) < target:
        raise RuntimeError(
            f"Collected only {len(events):,} valid interactions, below the "
            f"requested {target:,}. The subreddit listing is limited to the "
            f"newest {max_submissions:,} submissions and Reddit may not expose "
            "enough comments. Use --max-submissions 1000 if it is currently "
            "lower; otherwise choose another subreddit or retry later."
        )

    # Remove IDs assigned to deleted/invalid-only authors or empty posts so both
    # endpoint ID ranges remain contiguous for TGN's bipartite preprocessing.
    used_users = sorted({event["user_id"] for event in events})
    used_items = sorted({event["item_id"] for event in events})
    user_reindex = {old_id: new_id for new_id, old_id in enumerate(used_users, 1)}
    item_reindex = {old_id: new_id for new_id, old_id in enumerate(used_items, 1)}
    for event in events:
        event["user_id"] = user_reindex[event["user_id"]]
        event["item_id"] = item_reindex[event["item_id"]]

    events.sort(key=lambda event: event["timestamp"])
    return events


def save_datasets(events, output_dir, target):
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "reddit.csv"
    processed_path = output_dir / "ml_reddit.csv"
    edge_features_path = output_dir / "ml_reddit.npy"
    node_features_path = output_dir / "ml_reddit_node.npy"

    if len(events) != target:
        raise RuntimeError(
            f"Expected exactly {target:,} events before writing, got {len(events):,}."
        )

    feature_count = len(events[0]["features"])
    if any(len(event["features"]) != feature_count for event in events):
        raise ValueError("Interaction feature vectors have inconsistent dimensions.")

    with raw_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "user_id",
                "item_id",
                "timestamp",
                "state_label",
                "comma_separated_list_of_features",
            ]
        )
        for event in events:
            writer.writerow(
                [
                    event["user_id"],
                    event["item_id"],
                    event["timestamp"],
                    event["state_label"],
                    *event["features"],
                ]
            )

    raw = pd.DataFrame(
        {
            "u": [event["user_id"] for event in events],
            "i": [event["item_id"] for event in events],
            "ts": [event["timestamp"] for event in events],
            "label": [event["state_label"] for event in events],
            "idx": np.arange(len(events), dtype=np.int64),
        }
    )

    # Match the TGN preprocessing script's --bipartite reindexing convention.
    max_user_id = int(raw["u"].max())
    raw["i"] = raw["i"] + max_user_id + 1
    raw["u"] = raw["u"] + 1
    raw["i"] = raw["i"] + 1
    raw["idx"] = raw["idx"] + 1
    raw = raw[["u", "i", "ts", "label", "idx"]]
    raw.to_csv(processed_path, index=False)

    interaction_features = np.asarray(
        [event["features"] for event in events], dtype=np.float32
    )
    interaction_features = np.vstack(
        [np.zeros((1, feature_count), dtype=np.float32), interaction_features]
    )
    max_node_id = int(max(raw["u"].max(), raw["i"].max()))
    node_features = np.zeros((max_node_id + 1, feature_count), dtype=np.float32)
    np.save(edge_features_path, interaction_features)
    np.save(node_features_path, node_features)

    check = pd.read_csv(processed_path)
    if len(check) != target or list(check.columns) != [
        "u",
        "i",
        "ts",
        "label",
        "idx",
    ]:
        raise RuntimeError("Written TIGER dataset failed schema/count validation.")
    if not np.load(edge_features_path, mmap_mode="r").shape == (
        target + 1,
        feature_count,
    ):
        raise RuntimeError("Written edge features do not align with interactions.")

    return raw_path, processed_path, edge_features_path, node_features_path


def main():
    args = build_parser().parse_args()
    if args.target <= 0:
        raise ValueError("--target must be a positive integer.")
    if not 1 <= args.max_submissions <= MAX_SUBMISSIONS:
        raise ValueError(
            f"--max-submissions must be between 1 and {MAX_SUBMISSIONS}."
        )
    if args.max_more_per_post is not None and args.max_more_per_post < 0:
        raise ValueError("--max-more-per-post must be zero or greater.")

    reddit = praw.Reddit(
        client_id=required_credential("REDDIT_CLIENT_ID"),
        client_secret=required_credential("REDDIT_CLIENT_SECRET"),
        user_agent=required_credential("REDDIT_USER_AGENT"),
    )
    reddit.read_only = True

    print(
        f"Collecting up to {args.target:,} comment-to-post interactions "
        f"from r/{args.subreddit}..."
    )
    events = collect_interactions(
        reddit,
        args.subreddit,
        args.target,
        args.max_submissions,
        args.max_more_per_post,
    )
    paths = save_datasets(events, args.output_dir, args.target)
    print(f"Collected and validated {len(events):,} interactions.")
    for path in paths:
        print(f"Saved {path}")
    print(
        "TIGER can now load dataset 'reddit' with --root set to the project root."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Scraping failed: {error}", file=sys.stderr)
        raise
