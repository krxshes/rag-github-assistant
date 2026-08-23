"""
CLI entry point for the RAG GitHub assistant.

Usage:
    python src/cli.py                    # interactive mode
    python src/cli.py "your question"    # single-shot mode
"""

import sys

from agent import ask


BANNER = """
scikit-learn GitHub Issue Assistant
------------------------------------
Ask a troubleshooting question about scikit-learn. Answers are grounded in
real GitHub issues/comments, with citations verified against retrieved data.
Type 'exit' or Ctrl+C to quit.
"""


def print_result(result: dict):
    print(f"\n{result['answer']}\n")
    if result["cited_issues"]:
        status = "✓ verified" if result["fully_grounded"] else "⚠ some citations UNVERIFIED"
        print(f"[Citations: {result['cited_issues']} -- {status}]")
        if result["ungrounded"]:
            print(f"[Unverified: {result['ungrounded']} -- not found in retrieved sources]")
    print()


def interactive_loop():
    print(BANNER)
    while True:
        try:
            question = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            print("Goodbye.")
            break

        result = ask(question)
        print_result(result)


def main():
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        result = ask(question)
        print_result(result)
    else:
        interactive_loop()


if __name__ == "__main__":
    main()