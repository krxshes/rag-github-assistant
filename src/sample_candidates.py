"""
Helper: sample candidate issues to build eval questions from.

Usage:
    python src/sample_candidates.py

Picks issues that are:
- closed (more likely to have a clear resolution to ask about)
- have a reasonably long body (specific enough to write a real question)
- have at least one comment (so there's a resolution/discussion to draw on)

Prints title + first 200 chars of body + comment count, so you can quickly
scan and pick ~40 to write eval questions from.
"""

import json
import random

random.seed(42)

MIN_BODY_LEN = 150
N_SAMPLES = 60  # oversample; you'll hand-pick ~40 of these

candidates = []
with open("data/raw_issues.jsonl") as f:
    for line in f:
        issue = json.loads(line)
        if issue["state"] != "closed":
            continue
        if len(issue["body"]) < MIN_BODY_LEN:
            continue
        if len(issue["comments"]) < 1:
            continue
        candidates.append(issue)

print(f"Found {len(candidates)} eligible closed issues with substantial content.\n")

sample = random.sample(candidates, min(N_SAMPLES, len(candidates)))

for issue in sample:
    print(f"#{issue['number']} | {issue['title']}")
    print(f"  labels: {', '.join(issue['labels'])}")
    print(f"  body: {issue['body'][:200].strip().replace(chr(10), ' ')}...")
    print(f"  comments: {len(issue['comments'])} | url: {issue['html_url']}")
    print()