"""
Tests for the RAG GitHub assistant.

Usage:
    pytest tests/ -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from chunk import protect_code_blocks, restore_code_blocks, is_low_signal_comment, chunk_text
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --- chunking: code block protection ---

def test_code_block_survives_protection_roundtrip():
    text = "Here is a bug:\n```python\ndef f():\n    return None\n```\nThat's the issue."
    protected, blocks = protect_code_blocks(text)
    assert "```" not in protected  # code block replaced with placeholder
    restored = restore_code_blocks(protected, blocks)
    assert restored == text  # exact roundtrip, no data loss


def test_code_block_not_split_across_chunks():
    code = "```python\n" + "\n".join(f"line_{i} = {i}" for i in range(40)) + "\n```"
    text = f"Description text.\n\n{code}\n\nMore trailing text."
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    chunks = chunk_text(text, splitter)
    # the full code block should appear intact in exactly one chunk
    matches = [c for c in chunks if code in c]
    assert len(matches) == 1


def test_short_text_not_split():
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    text = "Short issue body."
    chunks = chunk_text(text, splitter)
    assert chunks == [text]


def test_empty_text_returns_no_chunks():
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    assert chunk_text("", splitter) == []
    assert chunk_text("   ", splitter) == []


# --- low-signal comment filtering ---

def test_bot_comment_is_filtered():
    comment = {"author": "codecov[bot]", "body": "Coverage report: 85% (+0.2%) this is a long enough body to pass length check"}
    assert is_low_signal_comment(comment) is True


def test_short_comment_is_filtered():
    comment = {"author": "someuser", "body": "+1"}
    assert is_low_signal_comment(comment) is True


def test_real_comment_is_kept():
    comment = {
        "author": "someuser",
        "body": "This can be fixed by increasing max_iter to 1000 and switching to the lbfgs solver.",
    }
    assert is_low_signal_comment(comment) is False


# --- citation verification ---

def test_verify_citations_all_grounded():
    from agent import verify_citations

    answer = "Fix this by upgrading, see Issue #123 and Issue #456 for details."
    steps = [
        (type("A", (), {"tool": "search_issues", "tool_input": "q"})(), "- Issue #123 [closed] (url): some text"),
        (type("A", (), {"tool": "search_issues", "tool_input": "q"})(), "- Issue #456 [open] (url): more text"),
    ]
    result = verify_citations(answer, steps)
    assert result["fully_grounded"] is True
    assert result["ungrounded"] == []
    assert result["grounded"] == [123, 456]


def test_verify_citations_detects_hallucination():
    from agent import verify_citations

    answer = "This was fixed in Issue #999, which was never actually retrieved."
    steps = [
        (type("A", (), {"tool": "search_issues", "tool_input": "q"})(), "- Issue #123 [closed] (url): some text"),
    ]
    result = verify_citations(answer, steps)
    assert result["fully_grounded"] is False
    assert result["ungrounded"] == [999]


def test_verify_citations_no_citations():
    from agent import verify_citations

    answer = "No relevant issues were found for this question."
    result = verify_citations(answer, [])
    assert result["fully_grounded"] is True
    assert result["cited_issues"] == []


# --- tool input parsing (quote stripping, malformed input handling) ---

def test_strip_quotes_removes_single_and_double():
    from agent import _strip_quotes

    assert _strip_quotes("'hello'") == "hello"
    assert _strip_quotes('"hello"') == "hello"
    assert _strip_quotes("hello") == "hello"
    assert _strip_quotes("  'hello'  ") == "hello"


def test_filter_by_label_handles_extra_pipe_segments():
    from agent import filter_by_label

    # model sometimes packs 'query|label|status' -- should still parse label correctly
    # and not crash, even though status gets ignored
    result = filter_by_label("NaN handling|Bug|open")
    assert isinstance(result, str)  # doesn't raise


def test_filter_by_status_rejects_invalid_status():
    from agent import filter_by_status

    result = filter_by_status("some query|maybe")
    assert "Error" in result


def test_filter_by_status_missing_pipe_returns_error():
    from agent import filter_by_status

    result = filter_by_status("some query with no pipe")
    assert "Error" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
