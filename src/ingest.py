"""
Step 1: Collect issues + comments from scikit-learn/scikit-learn via GitHub REST API.

Usage:
    Put GITHUB_TOKEN=ghp_xxx in a .env file at the project root, then:
    python src/ingest.py --n-issues 600 --out data/raw_issues.jsonl

Notes:
- Uses `state=all` so we get both open and closed issues (closed issues carry
  the resolution, which is the highest-value retrieval unit for a
  troubleshooting assistant).
- Filters out pull requests -- the GitHub REST API returns PRs in the
  /issues endpoint too, distinguished only by the presence of a
  `pull_request` key. We drop those.
- Checkpoints to disk incrementally (one JSON object per line) so a crash
  or rate-limit stall doesn't lose progress.
"""

import argparse
import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()  # reads GITHUB_TOKEN from a .env file in the project root

GITHUB_API = "https://api.github.com"
OWNER = "scikit-learn"
REPO = "scikit-learn"


def get_headers():
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError(
            "GITHUB_TOKEN not found. Add a line `GITHUB_TOKEN=ghp_xxx` to a "
            ".env file in the project root (a plain classic PAT with no "
            "scopes is enough for public repo read access)."
        )
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def check_rate_limit(resp):
    remaining = int(resp.headers.get("X-RateLimit-Remaining", 1))
    reset_at = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
    if remaining <= 1:
        sleep_for = max(reset_at - time.time(), 0) + 2
        print(f"  [rate limit] sleeping {sleep_for:.0f}s until reset...")
        time.sleep(sleep_for)


def fetch_issues(n_issues: int, headers: dict):
    """Paginate through /repos/{owner}/{repo}/issues, state=all, skipping PRs."""
    issues = []
    page = 1
    per_page = 100

    while len(issues) < n_issues:
        url = f"{GITHUB_API}/repos/{OWNER}/{REPO}/issues"
        params = {
            "state": "all",
            "per_page": per_page,
            "page": page,
            "sort": "updated",
            "direction": "desc",  # recent issues first: fresher, more likely resolved cleanly
        }
        resp = requests.get(url, headers=headers, params=params)
        resp.raise_for_status()
        check_rate_limit(resp)

        batch = resp.json()
        if not batch:
            break  # no more pages

        for item in batch:
            if "pull_request" in item:
                continue  # skip PRs
            issues.append(item)

        print(f"  page {page}: collected {len(issues)}/{n_issues} issues so far")
        page += 1

    return issues[:n_issues]


def fetch_comments(issue_number: int, headers: dict):
    url = f"{GITHUB_API}/repos/{OWNER}/{REPO}/issues/{issue_number}/comments"
    resp = requests.get(url, headers=headers, params={"per_page": 100})
    resp.raise_for_status()
    check_rate_limit(resp)
    return resp.json()


def structure_issue(raw_issue: dict, comments: list) -> dict:
    return {
        "number": raw_issue["number"],
        "title": raw_issue["title"],
        "body": raw_issue.get("body") or "",
        "state": raw_issue["state"],  # "open" | "closed"
        "labels": [l["name"] for l in raw_issue.get("labels", [])],
        "created_at": raw_issue["created_at"],
        "closed_at": raw_issue.get("closed_at"),
        "reaction_count": raw_issue.get("reactions", {}).get("total_count", 0),
        "comment_count": raw_issue.get("comments", 0),
        "html_url": raw_issue["html_url"],
        "comments": [
            {
                "author": c["user"]["login"] if c.get("user") else "unknown",
                "body": c.get("body") or "",
                "created_at": c["created_at"],
                "reaction_count": c.get("reactions", {}).get("total_count", 0),
            }
            for c in comments
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-issues", type=int, default=600)
    parser.add_argument("--out", type=str, default="data/raw_issues.jsonl")
    args = parser.parse_args()

    headers = get_headers()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Fetching {args.n_issues} issues from {OWNER}/{REPO}...")
    raw_issues = fetch_issues(args.n_issues, headers)

    print(f"Fetching comments for {len(raw_issues)} issues...")
    with out_path.open("w") as f:
        for i, raw in enumerate(raw_issues, 1):
            comments = fetch_comments(raw["number"], headers) if raw.get("comments", 0) > 0 else []
            structured = structure_issue(raw, comments)
            f.write(json.dumps(structured) + "\n")

            if i % 25 == 0:
                print(f"  {i}/{len(raw_issues)} issues processed "
                      f"(latest: #{raw['number']} - {raw['title'][:60]})")

            time.sleep(0.1)  # gentle pacing even with a high rate limit budget

    print(f"Done. Wrote {len(raw_issues)} issues to {out_path}")


if __name__ == "__main__":
    main()
