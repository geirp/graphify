"""Regression: second-pass imported calls must retain collision-salted Python callers.

The primary AST extractor disambiguates `_main`/`main` declaration IDs (#3302),
but the import-aware symbol-resolution pass formerly reconstructed the unsalted
ID of `_main`, silently assigning its cross-file calls to `main`.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from graphify.build import build_from_json
from graphify.extract import extract


_TARGETS = "def alpha():\n    return 1\n\ndef beta():\n    return 2\n"


def _extract(tmp_path: Path, caller_source: str, *, warm: bool = False) -> dict:
    targets = tmp_path / "targets.py"
    cli = tmp_path / "cli.py"
    targets.write_text(_TARGETS, encoding="utf-8")
    cli.write_text(caller_source, encoding="utf-8")
    paths = [cli, targets]
    result = extract(
        paths, root=tmp_path, cache_root=tmp_path / "cache", parallel=False,
    )
    if warm:
        result = extract(
            paths, root=tmp_path, cache_root=tmp_path / "cache", parallel=False,
        )
    return result


def _by_label(result: dict, source_file: str) -> dict[str, str]:
    return {
        node["label"]: node["id"]
        for node in result["nodes"]
        if node.get("source_file") == source_file
    }


def _imported_calls(result: dict) -> list[tuple[str, str, str, str]]:
    """(caller id, target id, line, confidence) for calls into targets.py."""
    target_ids = set(_by_label(result, "targets.py").values())
    return [
        (
            edge["source"], edge["target"],
            edge.get("source_location"), edge.get("confidence"),
        )
        for edge in result["edges"]
        if edge.get("relation") == "calls"
        and edge.get("source_file") == "cli.py"
        and edge.get("target") in target_ids
    ]


@pytest.mark.parametrize("private_first", [True, False], ids=["private-first", "public-first"])
def test_cross_file_calls_keep_canonical_caller_id(tmp_path, monkeypatch, private_first):
    monkeypatch.setenv("GRAPHIFY_BACKEND", "none")
    if private_first:
        source = (
            "from targets import alpha, beta\n\n"
            "def _main():\n    return beta()\n\n"
            "def main():\n    return alpha()\n"
        )
        beta_line, alpha_line = "L4", "L7"
    else:
        source = (
            "from targets import alpha, beta\n\n"
            "def main():\n    return alpha()\n\n"
            "def _main():\n    return beta()\n"
        )
        alpha_line, beta_line = "L4", "L7"

    result = _extract(tmp_path, source)
    ids = _by_label(result, "cli.py")
    targets = _by_label(result, "targets.py")
    private_id = f"cli_main_{hashlib.sha1(b'_main').hexdigest()[:6]}"
    assert ids["main()"] == "cli_main"
    assert ids["_main()"] == private_id

    expected = {
        (ids["_main()"], targets["beta()"], beta_line, "EXTRACTED"),
        (ids["main()"], targets["alpha()"], alpha_line, "EXTRACTED"),
    }
    observed = _imported_calls(result)
    assert set(observed) == expected, ("imported call ownership", observed)
    assert len(observed) == len(expected), "unexpected duplicate call evidence"
    assert (ids["main()"], targets["beta()"], beta_line, "EXTRACTED") not in observed


def test_no_collision_leaves_imported_call_ownership_unchanged(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPHIFY_BACKEND", "none")
    source = (
        "from targets import alpha, beta\n\n"
        "def start():\n    return alpha()\n\n"
        "def finish():\n    return beta()\n"
    )
    result = _extract(tmp_path, source)
    ids = _by_label(result, "cli.py")
    targets = _by_label(result, "targets.py")
    assert set(_imported_calls(result)) == {
        (ids["start()"], targets["alpha()"], "L4", "EXTRACTED"),
        (ids["finish()"], targets["beta()"], "L7", "EXTRACTED"),
    }


def test_other_underscore_collision_cannot_borrow_public_caller(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPHIFY_BACKEND", "none")
    source = (
        "from targets import alpha, beta\n\n"
        "def _invoke_scheduler():\n    return beta()\n\n"
        "def invoke_scheduler():\n    return alpha()\n"
    )
    result = _extract(tmp_path, source)
    ids = _by_label(result, "cli.py")
    targets = _by_label(result, "targets.py")
    assert ids["_invoke_scheduler()"] != ids["invoke_scheduler()"]
    assert set(_imported_calls(result)) == {
        (ids["_invoke_scheduler()"], targets["beta()"], "L4", "EXTRACTED"),
        (ids["invoke_scheduler()"], targets["alpha()"], "L7", "EXTRACTED"),
    }


def test_no_public_name_salts_both_callers(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPHIFY_BACKEND", "none")
    source = (
        "from targets import alpha, beta\n\n"
        "def _invoke():\n    return beta()\n\n"
        "def __invoke():\n    return alpha()\n"
    )
    result = _extract(tmp_path, source)
    ids = _by_label(result, "cli.py")
    targets = _by_label(result, "targets.py")
    assert ids["_invoke()"] != ids["__invoke()"]
    assert ids["_invoke()"] != "cli_invoke"
    assert ids["__invoke()"] != "cli_invoke"
    assert set(_imported_calls(result)) == {
        (ids["_invoke()"], targets["beta()"], "L4", "EXTRACTED"),
        (ids["__invoke()"], targets["alpha()"], "L7", "EXTRACTED"),
    }


def test_warm_extract_keeps_caller_attribution(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPHIFY_BACKEND", "none")
    source = (
        "from targets import alpha, beta\n\n"
        "def _main():\n    return beta()\n\n"
        "def main():\n    return alpha()\n"
    )
    result = _extract(tmp_path, source, warm=True)
    ids = _by_label(result, "cli.py")
    targets = _by_label(result, "targets.py")
    assert set(_imported_calls(result)) == {
        (ids["_main()"], targets["beta()"], "L4", "EXTRACTED"),
        (ids["main()"], targets["alpha()"], "L7", "EXTRACTED"),
    }


def test_built_graph_retains_correct_caller_directions(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPHIFY_BACKEND", "none")
    source = (
        "from targets import alpha, beta\n\n"
        "def _main():\n    return beta()\n\n"
        "def main():\n    return alpha()\n"
    )
    result = _extract(tmp_path, source)
    ids = _by_label(result, "cli.py")
    targets = _by_label(result, "targets.py")
    graph = build_from_json(result, root=str(tmp_path), directed=True)
    built_calls = {
        (src, dst)
        for src, dst, attrs in graph.edges(data=True)
        if attrs.get("relation") == "calls"
        and src in {ids["main()"], ids["_main()"]}
        and dst in {targets["alpha()"], targets["beta()"]}
    }
    assert built_calls == {
        (ids["_main()"], targets["beta()"]),
        (ids["main()"], targets["alpha()"]),
    }
