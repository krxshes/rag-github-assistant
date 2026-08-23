"""
Step 3: Embed chunks and load them into a persistent ChromaDB collection.

Usage:
    python src/embed.py --in data/chunks.jsonl --db-path ./chroma_db

Notes:
- Uses sentence-transformers 'all-MiniLM-L6-v2' -- local, free, no API cost,
  and fast enough for ~4-8k chunks on a laptop CPU. Defensible in an
  interview since it's not just "I called the OpenAI embeddings endpoint."
- Chroma metadata values must be str/int/float/bool (no None, no lists),
  so we sanitize before insert -- this is a common gotcha.
- Batched adds (not one-by-one) since Chroma/embedding throughput is much
  better in batches, and 4k+ single calls would be slow.
"""

import argparse
import json
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm

BATCH_SIZE = 128


def sanitize_metadata(chunk: dict) -> dict:
    """Chroma metadata values must be str, int, float, or bool -- no None/lists."""
    return {
        "issue_number": int(chunk["issue_number"]),
        "title": chunk["title"] or "",
        "state": chunk["state"] or "unknown",
        "labels": chunk["labels"] or "",          # already comma-joined string from chunk.py
        "url": chunk["url"] or "",
        "reaction_count": int(chunk["reaction_count"] or 0),
        "reaction_count_source": int(chunk["reaction_count_source"] or 0),
        "is_comment": bool(chunk["is_comment"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", type=str, default="data/chunks.jsonl")
    parser.add_argument("--db-path", type=str, default="./chroma_db")
    parser.add_argument("--collection", type=str, default="github_issues")
    args = parser.parse_args()

    print("Loading chunks...")
    chunks = []
    with open(args.in_path) as f:
        for line in f:
            chunks.append(json.loads(line))
    print(f"  {len(chunks)} chunks loaded")

    print("Setting up embedding function (BAAI/bge-small-en-v1.5, first run downloads the model)...")
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-small-en-v1.5")

    client = chromadb.PersistentClient(path=args.db_path)

    # Drop any existing collection with the same name so reruns are clean
    # instead of silently duplicating/erroring on existing IDs.
    existing = [c.name for c in client.list_collections()]
    if args.collection in existing:
        print(f"Collection '{args.collection}' already exists -- deleting to rebuild fresh.")
        client.delete_collection(args.collection)

    collection = client.create_collection(name=args.collection, embedding_function=ef)

    print(f"Embedding + inserting {len(chunks)} chunks in batches of {BATCH_SIZE}...")
    for i in tqdm(range(0, len(chunks), BATCH_SIZE)):
        batch = chunks[i:i + BATCH_SIZE]
        documents = [c["text"] for c in batch]
        ids = [c["chunk_id"] for c in batch]
        metadatas = [sanitize_metadata(c) for c in batch]

        collection.add(documents=documents, ids=ids, metadatas=metadatas)

    print(f"Done. Collection '{args.collection}' now has {collection.count()} chunks.")
    print(f"Persisted at {args.db_path}")


if __name__ == "__main__":
    main()
