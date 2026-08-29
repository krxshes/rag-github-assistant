"""
MCP server exposing the scikit-learn issue index.

Runs the same retrieval tools as agent.py over MCP, so any MCP client
(Claude Desktop, Cline, etc.) can query the index directly without going
through the local ReAct loop.

Exposes:
  Tools     - search_issues, filter_by_status, filter_by_label,
              rank_by_popularity, verify_citations
  Resources - the collection itself, plus individual issues by number
  Prompts   - troubleshoot, find_similar

Usage:
    python src/mcp_server.py

Register with Claude Desktop by adding to claude_desktop_config.json:
    {
      "mcpServers": {
        "sklearn-issues": {
          "command": "/absolute/path/to/venv/bin/python",
          "args": ["/absolute/path/to/src/mcp_server.py"]
        }
      }
    }
"""

import json

from mcp.server.fastmcp import FastMCP

from agent import (
    _collection,
    _query_chroma,
    _dedupe_by_issue,
    _format_results,
    filter_by_label,
    filter_by_status,
    rank_by_popularity,
    search_issues,
    _extract_cited_issue_numbers,
)

mcp = FastMCP("sklearn-issues")


# --- tools -------------------------------------------------------------

@mcp.tool()
def search(query: str) -> str:
    """Semantic search over scikit-learn GitHub issues and comments.

    Use this as the default tool for most questions. Returns up to 5
    issues, deduplicated so one long thread cannot dominate results.
    """
    return search_issues(query)


@mcp.tool()
def search_by_status(query: str, status: str) -> str:
    """Search issues filtered to open or closed state.

    Use closed for "how was this resolved"; open for current unresolved
    problems. status must be exactly 'open' or 'closed'.
    """
    return filter_by_status(f"{query}|{status}")


@mcp.tool()
def search_by_label(query: str, label: str) -> str:
    """Search issues carrying a given GitHub label.

    Common labels: Bug, Enhancement, Documentation, Performance, API.
    """
    return filter_by_label(f"{query}|{label}")


@mcp.tool()
def search_by_popularity(query: str) -> str:
    """Search issues and rank by community reaction count.

    Use for "most common", "most upvoted", or "most discussed" questions.
    """
    return rank_by_popularity(query)


@mcp.tool()
def check_citations(answer: str, retrieved_text: str) -> str:
    """Verify that every issue number cited in an answer appears in the
    retrieved source text.

    Returns a JSON report listing grounded and ungrounded citations. Use
    this to catch hallucinated issue references before presenting an answer.
    """
    cited = _extract_cited_issue_numbers(answer)
    retrieved = _extract_cited_issue_numbers(retrieved_text)
    ungrounded = sorted(cited - retrieved)
    return json.dumps({
        "cited": sorted(cited),
        "grounded": sorted(cited & retrieved),
        "ungrounded": ungrounded,
        "fully_grounded": not ungrounded,
    }, indent=2)


# --- resources ---------------------------------------------------------

@mcp.resource("sklearn://collection/stats")
def collection_stats() -> str:
    """Size and configuration of the indexed issue collection."""
    return json.dumps({
        "collection": "github_issues",
        "chunk_count": _collection.count(),
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "source_repo": "scikit-learn/scikit-learn",
    }, indent=2)


@mcp.resource("sklearn://issue/{number}")
def issue_by_number(number: str) -> str:
    """All indexed chunks for a single issue, by issue number."""
    res = _collection.get(where={"issue_number": int(number)})
    if not res["ids"]:
        return f"No indexed content for issue #{number}."
    meta = res["metadatas"][0]
    parts = [
        f"Issue #{meta['issue_number']} [{meta['state']}]",
        f"Title: {meta['title']}",
        f"Labels: {meta['labels'] or 'none'}",
        f"Reactions: {meta['reaction_count']}",
        f"URL: {meta['url']}",
        "",
    ]
    parts += res["documents"]
    return "\n".join(parts)


# --- prompts -----------------------------------------------------------

@mcp.prompt()
def troubleshoot(error: str) -> str:
    """Diagnose a scikit-learn error against real reported issues."""
    return (
        f"I'm hitting this error in scikit-learn:\n\n{error}\n\n"
        "Search the issue index for related reports. Prefer closed issues, "
        "since those carry resolutions. Cite the issue number and URL for "
        "every claim, then run check_citations on your answer before "
        "presenting it."
    )


@mcp.prompt()
def find_similar(description: str) -> str:
    """Find existing issues similar to a bug you're about to report."""
    return (
        f"Before I file a new scikit-learn issue, check whether this has "
        f"already been reported:\n\n{description}\n\n"
        "Search both open and closed issues, and rank by reaction count to "
        "surface the most-discussed duplicates. Report whether this looks "
        "like a duplicate and of what."
    )


if __name__ == "__main__":
    mcp.run()