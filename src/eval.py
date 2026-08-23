"""
Step 6: Evaluate retrieval accuracy against the curated eval benchmark.

Usage:
    python src/eval.py --k 5

Computes hits@k: for each question, does the top-k retrieved set contain at
least one chunk from the ground-truth issue? Reports the overall percentage,
plus per-question pass/fail so you can see exactly what's weak.
"""

import argparse
import json

import chromadb
from chromadb.utils import embedding_functions

DB_PATH = "./chroma_db"
COLLECTION_NAME = "github_issues"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=str, default="data/eval_questions.json")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    with open(args.questions) as f:
        questions = json.load(f)

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-small-en-v1.5")
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(COLLECTION_NAME, embedding_function=ef)

    hits = 0
    results_log = []

    for item in questions:
        query = BGE_QUERY_PREFIX + item["question"]
        expected = set(item["expected_issue_numbers"])

        raw = collection.query(query_texts=[query], n_results=args.k * 4)  # overfetch, then dedupe by issue
        retrieved_issue_numbers = []
        seen = set()
        for meta in raw["metadatas"][0]:
            num = meta["issue_number"]
            if num not in seen:
                seen.add(num)
                retrieved_issue_numbers.append(num)
            if len(retrieved_issue_numbers) >= args.k:
                break

        hit = bool(expected & set(retrieved_issue_numbers))
        hits += hit
        results_log.append({
            "question": item["question"],
            "expected": sorted(expected),
            "retrieved_top_k": retrieved_issue_numbers,
            "hit": hit,
        })

    accuracy = hits / len(questions) if questions else 0.0

    print(f"\n{'='*70}")
    print(f"RETRIEVAL ACCURACY (hits@{args.k}): {hits}/{len(questions)} = {accuracy:.1%}")
    print(f"{'='*70}\n")

    print("Misses (worth reviewing):\n")
    for r in results_log:
        if not r["hit"]:
            print(f"  Q: {r['question']}")
            print(f"     expected: {r['expected']} | got: {r['retrieved_top_k']}\n")

    out_path = "data/eval_results.json"
    with open(out_path, "w") as f:
        json.dump({"k": args.k, "accuracy": accuracy, "hits": hits, "total": len(questions),
                    "results": results_log}, f, indent=2)
    print(f"Full results written to {out_path}")


if __name__ == "__main__":
    main()