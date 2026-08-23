"""
Step 4: Agentic tool-use over the vector store via a ReAct reasoning loop.

Usage:
    python src/agent.py

Notes:
- Runs fully locally via Ollama (llama3.1:8b) -- no API cost.
- Hand-written ReAct prompt (not the LangChain hub default) so the exact
  reasoning format is visible/explainable, and so we can tune it hard
  against local-model format drift (missing "Action Input:", extra prose,
  hallucinated tool names -- the main failure modes with 8B models).
- Four tools:
    1. search_issues        -- baseline semantic search
    2. filter_by_status     -- semantic search + state=open/closed filter
    3. filter_by_label      -- semantic search + label substring filter
    4. rank_by_popularity   -- wider semantic candidate pool, re-sorted by
                                reaction_count metadata
- All tools de-duplicate by issue_number in their results (max 1 chunk per
  issue in the returned set) so one long thread can't dominate results --
  the mitigation we discussed after seeing a few mega-threads in the data.
- BGE query prefix ("Represent this sentence for searching relevant
  passages: ") is applied inside every tool automatically, so the agent
  and any caller never has to remember to add it.
"""

import chromadb
from chromadb.utils import embedding_functions
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

import re


def _extract_cited_issue_numbers(text: str) -> set:
    """Find all issue numbers referenced in text, matching '#1234' or 'Issue #1234'."""
    return {int(n) for n in re.findall(r"#(\d+)", text)}


def _extract_retrieved_issue_numbers(steps) -> set:
    """Find all issue numbers that actually appeared in tool Observations."""
    numbers = set()
    for _action, observation in steps:
        numbers |= _extract_cited_issue_numbers(str(observation))
    return numbers


def verify_citations(answer: str, steps) -> dict:
    """
    Check every issue number cited in the final answer against what was
    actually retrieved. Returns a dict with the answer plus a grounding report.
    """
    cited = _extract_cited_issue_numbers(answer)
    retrieved = _extract_retrieved_issue_numbers(steps)
    ungrounded = cited - retrieved

    return {
        "answer": answer,
        "cited_issues": sorted(cited),
        "grounded": sorted(cited & retrieved),
        "ungrounded": sorted(ungrounded),
        "fully_grounded": len(ungrounded) == 0,
    }

DB_PATH = "./chroma_db"
COLLECTION_NAME = "github_issues"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

def _strip_quotes(s: str) -> str:
    """Local models often wrap Action Input in quotes -- strip them so tool parsing works."""
    return s.strip().strip("'\"").strip()

# --- vector store setup ---
_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-small-en-v1.5")
_client = chromadb.PersistentClient(path=DB_PATH)
_collection = _client.get_collection(COLLECTION_NAME, embedding_function=_ef)


def _dedupe_by_issue(chunks: list[dict], k: int) -> list[dict]:
    """Keep at most one chunk per issue_number, preserving order (best match first)."""
    seen = set()
    out = []
    for c in chunks:
        num = c["metadata"]["issue_number"]
        if num in seen:
            continue
        seen.add(num)
        out.append(c)
        if len(out) >= k:
            break
    return out


def _query_chroma(query: str, n_results: int, where: dict | None = None) -> list[dict]:
    prefixed = BGE_QUERY_PREFIX + query
    results = _collection.query(
        query_texts=[prefixed],
        n_results=n_results,
        where=where,
    )
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    return [{"text": d, "metadata": m} for d, m in zip(docs, metas)]


def _format_results(chunks: list[dict]) -> str:
    if not chunks:
        return "No matching issues found."
    lines = []
    for c in chunks:
        m = c["metadata"]
        snippet = c["text"][:300].replace("\n", " ")
        lines.append(
            f"- Issue #{m['issue_number']} [{m['state']}] ({m['url']}): {snippet}..."
        )
    return "\n".join(lines)


# --- tool implementations ---

def search_issues(query: str) -> str:
    """Baseline semantic search over all issue/comment chunks."""
    query = _strip_quotes(query)
    raw = _query_chroma(query, n_results=15)
    top = _dedupe_by_issue(raw, k=5)
    return _format_results(top)


def filter_by_status(query_and_status: str) -> str:
    """Input format: 'query|status' where status is 'open' or 'closed'."""
    query_and_status = _strip_quotes(query_and_status)
    if "|" not in query_and_status:
        return "Error: input must be formatted as 'query|status' (status: open or closed)."
    query, status = query_and_status.split("|", 1)
    query, status = _strip_quotes(query), _strip_quotes(status).lower()
    if status not in ("open", "closed"):
        return "Error: status must be exactly 'open' or 'closed'."

    raw = _query_chroma(query, n_results=20, where={"state": status})
    top = _dedupe_by_issue(raw, k=5)
    return _format_results(top)


def filter_by_label(query_and_label: str) -> str:
    """Input format: 'query|label', e.g. 'convergence warning|bug'."""
    query_and_label = _strip_quotes(query_and_label)
    parts = [p.strip() for p in query_and_label.split("|") if p.strip()]
    if len(parts) < 2:
        return "Error: input must be formatted as 'query|label'."
    query, label = parts[0], parts[1].lower()  # ignore any extra segments beyond label

    raw = _query_chroma(query, n_results=40)
    filtered = [c for c in raw if label in c["metadata"]["labels"].lower()]
    top = _dedupe_by_issue(filtered, k=5)
    if not top:
        return f"No issues found matching label '{label}'. Try search_issues without a label filter."
    return _format_results(top)


def rank_by_popularity(query: str) -> str:
    """Semantic search over a wide candidate pool, re-sorted by reaction count."""
    query = _strip_quotes(query)
    raw = _query_chroma(query, n_results=30)
    deduped = _dedupe_by_issue(raw, k=15)
    ranked = sorted(deduped, key=lambda c: c["metadata"]["reaction_count"], reverse=True)
    top = ranked[:5]
    if not top:
        return "No matching issues found."
    lines = []
    for c in top:
        m = c["metadata"]
        snippet = c["text"][:300].replace("\n", " ")
        lines.append(
            f"- Issue #{m['issue_number']} [{m['reaction_count']} reactions] "
            f"({m['url']}): {snippet}..."
        )
    return "\n".join(lines)


TOOLS = [
    Tool(
        name="search_issues",
        func=search_issues,
        description=(
            "Semantic search over scikit-learn GitHub issues and comments. "
            "Input: a plain query string describing the problem or topic. "
            "Use this as your default/first tool for most questions."
        ),
    ),
    Tool(
        name="filter_by_status",
        func=filter_by_status,
        description=(
            "Search issues filtered by state. Use when the user wants to know "
            "how something WAS resolved (search closed issues) or wants only "
            "currently open/unresolved issues. "
            "Input MUST be formatted exactly as 'query|status' where status is "
            "'open' or 'closed', e.g. 'ConvergenceWarning logistic regression|closed'."
        ),
    ),
    Tool(
    name="filter_by_label",
    func=filter_by_label,
    description=(
        "Search issues filtered by a GitHub label such as 'Bug', 'Enhancement', "
        "'Documentation', 'Performance', or 'API'. Input MUST be formatted as "
        "'query|label' with EXACTLY one pipe -- do not add a status here. "
        "e.g. 'GridSearchCV memory|Bug'. Use filter_by_status separately if "
        "you also need to filter by open/closed."
        ),
    ),
    Tool(
        name="rank_by_popularity",
        func=rank_by_popularity,
        description=(
            "Search issues and rank results by community reaction count "
            "(thumbs up/etc). Use when the user asks for the 'most common', "
            "'most upvoted', or 'most discussed' issues on a topic. "
            "Input: a plain query string."
        ),
    ),
]


# --- hand-written ReAct prompt, tuned for local 8B model reliability ---
REACT_PROMPT = PromptTemplate.from_template(
    """You are a troubleshooting assistant for the scikit-learn GitHub repository.
You answer developer questions by searching real GitHub issues and comments,
and you always cite the issue number and URL for any claim you make.

You have access to the following tools:

{tools}

Use this format. Follow the EXAMPLE below exactly -- do not repeat the word
"Question" after the first line, and never invent a new Question yourself.

EXAMPLE:
Question: How do I fix a memory leak in Pipeline?
Thought: I should search for issues about Pipeline memory leaks.
Action: search_issues
Action Input: Pipeline memory leak
Observation: - Issue #111 [closed] (https://github.com/.../111): use joblib caching...
Thought: I now know the final answer.
Final Answer: You can fix this by enabling joblib caching, as shown in Issue #111 (https://github.com/.../111).

Now do the same for the real question below, using only these tools: [{tool_names}].
Stop immediately after writing one Action Input -- do not write your own
Observation, do not write a new Question.

Question: {input}
Thought:{agent_scratchpad}
If your next Action and Action Input would be IDENTICAL to your previous one,
do not repeat it. Instead, write "Thought: I now know the final answer" and
give a Final Answer using the best information already gathered, even if
imperfect -- explicitly note if the fix wasn't clearly confirmed in the results."""

)

def build_agent_executor():
    llm = ChatOllama(
        model="llama3.1:8b",
        temperature=0,
        stop=["\nObservation:", "\nObservation"],
    )
    agent = create_react_agent(llm, TOOLS, REACT_PROMPT)
    return AgentExecutor(
        agent=agent,
        tools=TOOLS,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=4,
        early_stopping_method="force",
        return_intermediate_steps=True,
    )

def ask(question: str) -> dict:
    executor = build_agent_executor()
    result = executor.invoke({"input": question})
    output = result["output"]
    steps = result.get("intermediate_steps", [])

    if "stopped due to" in output.lower() or not output.strip():
        if not steps:
            return {
                "answer": "No relevant issues were found for this question.",
                "cited_issues": [], "grounded": [], "ungrounded": [], "fully_grounded": True,
            }

        observations = "\n\n".join(
            f"[{action.tool} -> {action.tool_input}]\n{observation}"
            for action, observation in steps
        )
        llm = ChatOllama(model="llama3.1:8b", temperature=0)
        synthesis_prompt = (
            "Based ONLY on the following search results from scikit-learn "
            "GitHub issues, answer the question. Cite issue numbers and URLs "
            "exactly as they appear below. If the results don't clearly answer "
            "the question, say so explicitly.\n\n"
            f"Question: {question}\n\n"
            f"Search results:\n{observations}\n\n"
            "Answer:"
        )
        output = llm.invoke(synthesis_prompt).content

    return verify_citations(output, steps)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "How do I fix a ConvergenceWarning in LogisticRegression?"

    print(f"\nQUESTION: {question}\n")
    result = ask(question)
    print(f"\nFINAL ANSWER:\n{result['answer']}")
    print(f"\n--- Citation grounding check ---")
    print(f"Cited issues: {result['cited_issues']}")
    print(f"Grounded (verified in retrieved data): {result['grounded']}")
    if result["ungrounded"]:
        print(f"⚠ UNGROUNDED (hallucinated, not in retrieved data): {result['ungrounded']}")
    else:
        print("✓ All citations verified against retrieved data.")