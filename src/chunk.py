"""
Step 2: Chunk structured issues into retrieval units.

Usage:
    python src/chunk.py --in data/raw_issues.jsonl --out data/chunks.jsonl

Strategy:
- Each issue body and each comment is chunked SEPARATELY, not concatenated.
  This means a single accepted-answer comment can be retrieved on its own
  with high precision, rather than being buried inside a giant issue blob.
- Code blocks (```...```) and traceback-looking blocks are protected from
  mid-block splitting -- sklearn issues are full of these, and splitting a
  traceback in half destroys its meaning for retrieval.
- Short comments (under the chunk size) stay as a single chunk untouched.
- Bot comments (CI/coverage bots) and very short low-signal comments
  ("+1", "same issue") are filtered out entirely -- they add volume but
  no retrieval value, and dilute mega-threads even further.
- Every chunk carries metadata needed by the agentic tools later: issue
  number, title, state, labels, url, reaction_count, is_comment.
"""

import argparse
import json
import re
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100

CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)

# Bot accounts that post low-signal automated comments (CI status, coverage
# reports, stale-issue pings) -- these add volume but no retrieval value.
BOT_AUTHORS = {
    "codecov", "codecov[bot]", "github-actions", "github-actions[bot]",
    "lgtm-com[bot]", "sklearn-lgtm", "azure-pipelines[bot]", "welcome[bot]",
    "stale[bot]",
}

# Very short comments like "+1", "same here", "ping" carry no troubleshooting
# signal and just add noise/volume to the chunk corpus.
MIN_COMMENT_CHARS = 40


def is_low_signal_comment(comment: dict) -> bool:
    author = (comment.get("author") or "").lower()
    if author in BOT_AUTHORS:
        return True
    body = (comment.get("body") or "").strip()
    if len(body) < MIN_COMMENT_CHARS:
        return True
    return False


def protect_code_blocks(text: str):
    """
    Replace code blocks with placeholders before splitting, so the splitter
    never breaks a code block across two chunks. Returns the placeholder'd
    text plus a dict to restore the real code afterward.
    """
    blocks = {}

    def _replace(match):
        idx = len(blocks)
        placeholder = f"__CODEBLOCK_{idx}__"
        blocks[placeholder] = match.group(0)
        return placeholder

    protected_text = CODE_BLOCK_RE.sub(_replace, text)
    return protected_text, blocks


def restore_code_blocks(text: str, blocks: dict) -> str:
    for placeholder, code in blocks.items():
        text = text.replace(placeholder, code)
    return text


def chunk_text(text: str, splitter: RecursiveCharacterTextSplitter) -> list[str]:
    if not text or not text.strip():
        return []

    protected, blocks = protect_code_blocks(text)

    if len(protected) <= CHUNK_SIZE:
        return [restore_code_blocks(protected, blocks)]

    raw_chunks = splitter.split_text(protected)
    return [restore_code_blocks(c, blocks) for c in raw_chunks]


def build_chunks(issue: dict, splitter: RecursiveCharacterTextSplitter) -> list[dict]:
    chunks = []
    base_meta = {
        "issue_number": issue["number"],
        "title": issue["title"],
        "state": issue["state"],
        "labels": ",".join(issue["labels"]) if issue["labels"] else "",
        "url": issue["html_url"],
        "reaction_count": issue["reaction_count"],
    }

    # --- issue body ---
    body_text = f"{issue['title']}\n\n{issue['body']}"
    for i, piece in enumerate(chunk_text(body_text, splitter)):
        chunks.append({
            **base_meta,
            "chunk_id": f"{issue['number']}-body-{i}",
            "text": piece,
            "is_comment": False,
            "reaction_count_source": issue["reaction_count"],
        })

    # --- comments (each chunked independently) ---
    for c_idx, comment in enumerate(issue["comments"]):
        if is_low_signal_comment(comment):
            continue
        for i, piece in enumerate(chunk_text(comment["body"], splitter)):
            chunks.append({
                **base_meta,
                "chunk_id": f"{issue['number']}-comment{c_idx}-{i}",
                "text": piece,
                "is_comment": True,
                "reaction_count_source": comment["reaction_count"],
            })

    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", type=str, default="data/raw_issues.jsonl")
    parser.add_argument("--out", type=str, default="data/chunks.jsonl")
    args = parser.parse_args()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    in_path = Path(args.in_path)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_chunks = 0
    total_issues = 0
    empty_bodies = 0
    filtered_comments = 0
    kept_comments = 0

    with in_path.open() as f_in, out_path.open("w") as f_out:
        for line in f_in:
            issue = json.loads(line)
            total_issues += 1

            if not issue["body"].strip():
                empty_bodies += 1

            for comment in issue["comments"]:
                if is_low_signal_comment(comment):
                    filtered_comments += 1
                else:
                    kept_comments += 1

            for chunk in build_chunks(issue, splitter):
                f_out.write(json.dumps(chunk) + "\n")
                total_chunks += 1

    print(f"Processed {total_issues} issues -> {total_chunks} chunks")
    print(f"  (avg {total_chunks / max(total_issues,1):.1f} chunks/issue)")
    print(f"  {empty_bodies} issues had an empty body (comment-only chunks used)")
    print(f"  filtered {filtered_comments} low-signal comments (bots/short), kept {kept_comments}")
    print(f"Wrote to {out_path}")


if __name__ == "__main__":
    main()