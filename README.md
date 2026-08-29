# scikit-learn GitHub Issue Assistant

A retrieval-augmented, agentic troubleshooting assistant built over real
scikit-learn GitHub issues. Runs **fully locally** (Ollama + local
embeddings) with zero API cost, using a ReAct reasoning loop with four
purpose-built tools and citation-grounded, hallucination-checked answers.

A retrieval-augmented, agentic troubleshooting assistant built over real
scikit-learn GitHub issues. Runs **fully locally** (Ollama + local
embeddings) with zero API cost, using a ReAct reasoning loop with four
purpose-built tools and citation-grounded, hallucination-checked answers.

![Demo](demo.gif)

## What it does

## What it does

Ask a question like *"How do I fix a ConvergenceWarning in
LogisticRegression?"* and the assistant:

1. Reasons step-by-step (ReAct: Thought → Action → Observation) about which
   tool to use
2. Retrieves relevant scikit-learn issue/comment chunks via semantic search
   over a local vector store
3. Optionally filters by issue status (open/closed), label, or ranks by
   community reaction count, depending on the question
4. Synthesizes an answer citing real issue numbers and URLs
5. **Verifies** every cited issue number actually appeared in retrieved
   data before returning the answer -- catching hallucinated citations
   rather than trusting the LLM's discipline

## Architecture

```
GitHub REST API  ->  ingest.py   ->  raw_issues.jsonl  (600 issues, scikit-learn)
                      chunk.py    ->  chunks.jsonl      (body/comment split, code-block-safe, bot-filtered)
                      embed.py    ->  ChromaDB           (BAAI/bge-small-en-v1.5, local)
                      agent.py    ->  ReAct loop (Ollama llama3.1:8b) + 4 tools + citation verification
                      eval.py     ->  retrieval accuracy benchmark
                      cli.py      ->  interactive / single-shot CLI
```

**Tools available to the agent:**
| Tool | Purpose |
|---|---|
| `search_issues` | baseline semantic search |
| `filter_by_status` | semantic search restricted to open/closed issues |
| `filter_by_label` | semantic search restricted to a GitHub label |
| `rank_by_popularity` | semantic search re-ranked by reaction count |

All tools de-duplicate by issue number so a single long discussion thread
can't dominate results.

## Eval results

Retrieval accuracy measured on a hand-curated 45-question benchmark
(`data/eval_questions.json`), including 8 deliberately ambiguous/overlapping
questions (e.g. multiple issues discussing NaN handling, ConvergenceWarning,
GridSearchCV memory) to stress-test retrieval beyond easy lookup cases.

| Metric | Score |
|---|---|
| hits@1 | **95.6%** (43/45) |
| hits@5 | 100% (45/45) |

hits@1 is the more meaningful number here -- it requires the single top
retrieved result to be correct, not just "somewhere in the top 5." Run
`python src/eval.py --k 1` to reproduce.

**Known limitations (found and documented, not hidden):**
- Narrow, jargon-heavy technical queries (e.g. `feature_importances_`
  correctness) retrieve less precisely than broader topic queries -- a
  known characteristic of small local embedding models.
- The local 8B LLM occasionally repeats an identical tool call before
  converging, or fails to produce a clean `Final Answer` within the
  iteration budget. When this happens, the system falls back to a direct
  synthesis call over whatever was actually retrieved, rather than
  dead-ending -- see `ask()` in `src/agent.py`.
- A handful of very old, long-running discussion issues (100+ comments)
  exist in the corpus; tool-level de-duplication by issue number prevents
  them from dominating results.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Ollama (local LLM, no API cost)
brew install ollama
ollama serve &
ollama pull llama3.1:8b

# GitHub token for data collection (classic PAT, no scopes needed for public repos)
echo "GITHUB_TOKEN=ghp_xxx" > .env
```

## Usage

```bash
# 1. Collect issues from scikit-learn/scikit-learn
python src/ingest.py --n-issues 600 --out data/raw_issues.jsonl

# 2. Chunk into retrieval units
python src/chunk.py --in data/raw_issues.jsonl --out data/chunks.jsonl

# 3. Embed + load into ChromaDB
python src/embed.py --in data/chunks.jsonl --db-path ./chroma_db

# 4. Ask questions
python src/cli.py "How do I fix a ConvergenceWarning in LogisticRegression?"
python src/cli.py   # interactive mode

# 5. Run the eval benchmark
python src/eval.py --k 1
```
## MCP interface

The same retrieval tools are exposed over the [Model Context Protocol](https://modelcontextprotocol.io),
so any MCP client can query the issue index directly without going through
the local ReAct loop.

**Tools**

| Tool | Purpose |
|---|---|
| `search` | Semantic search over issues and comments |
| `search_by_status` | Same, filtered to open or closed |
| `search_by_label` | Same, filtered by GitHub label |
| `search_by_popularity` | Ranked by community reaction count |
| `check_citations` | Verify cited issue numbers against retrieved text |

**Resources**

- `sklearn://collection/stats` — index size and embedding model
- `sklearn://issue/{number}` — all indexed chunks for one issue

**Prompts**

- `troubleshoot` — diagnose an error against reported issues
- `find_similar` — check whether a bug is already reported

Run the server:

    python src/mcp_server.py

Inspect it interactively:

    npx @modelcontextprotocol/inspector venv/bin/python src/mcp_server.py

Register with Claude Desktop by adding to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "sklearn-issues": {
      "command": "/absolute/path/to/rag-github-assistant/venv/bin/python",
      "args": ["/absolute/path/to/rag-github-assistant/src/mcp_server.py"]
    }
  }
}
```

Note that the MCP tools take separate typed arguments rather than the
pipe-delimited strings used internally by the ReAct agent — that format
existed to work around format drift in local 8B models, which MCP clients
don't have.

## Tests

```bash
pytest tests/ -v
```

Covers code-block-safe chunking, bot/low-signal comment filtering, citation
grounding and hallucination detection, and tool input robustness (quote
stripping, malformed input handling).

## Stack

Python, LangChain (ReAct agent), ChromaDB, sentence-transformers
(`BAAI/bge-small-en-v1.5`), Ollama (`llama3.1:8b`), GitHub REST API, pytest.