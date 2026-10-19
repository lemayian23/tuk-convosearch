"""
Evaluation harness.

    python -m evals.runner                    # TU-K student questions
    python -m evals.runner cancer             # PubMed questions
    python -m evals.runner tuk llm            # with the LLM critic
    python -m evals.runner tuk heuristic false  # no query expansion

Needs the models in TUK_OLLAMA_MODEL to be pulled and the TU-K corpus
indexed (scripts/ingest_corpus.py), so this is an integration run rather
than part of the unit suite.

Scoring is deliberately unforgiving about the two things that make a student
assistant dangerous: answering from outside the documents, and answering a
question the router should have refused. Those are the `refusal` and
`router_control` categories.

Location: evals/runner.py
"""

import io
import sys
import time
from typing import Any, Dict, List

# Force UTF-8 stdout/stderr on Windows so redirected output doesn't
# crash on ✓/✗ characters. Also silences the copy-paste mojibake.
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from app.rag import RAGContext

REFUSAL_MARKERS = (
    "cannot find",
    "can not find",
    "could not find",
    "not find this",
    "do not have",
    "don't have",
    "not stated",
    "not specified",
    "not been specified",
    "not mentioned",
    "not available",
    "not address",
    "does not specify",
    "does not mention",
)


def build_pipeline(name: str, critic_mode: str, enable_expansion: bool):
    if name == "tuk":
        from app.rag.configs.tuk_pipeline import build_tuk_pipeline

        return build_tuk_pipeline(critic_mode=critic_mode)
    if name == "cancer":
        from app.rag.configs.cancer_pipeline import build_cancer_pipeline

        return build_cancer_pipeline(
            critic_mode=critic_mode, enable_expansion=enable_expansion
        )
    raise SystemExit(f"Unknown pipeline {name!r}; expected 'tuk' or 'cancer'")


def questions_for(name: str) -> List[Dict[str, Any]]:
    if name == "tuk":
        from evals.questions import TUK_EVAL_QUESTIONS

        return TUK_EVAL_QUESTIONS
    from evals.questions import EVAL_QUESTIONS

    return EVAL_QUESTIONS


# ---------------------------------------------------------------------- #
# Scoring
# ---------------------------------------------------------------------- #

def score_question(ctx: RAGContext, expected: Dict[str, Any]) -> Dict[str, Any]:
    """Score a single pipeline run against its expected outcome."""
    expected_route = expected.get("expected_route", "internal_docs")
    route_ok = ctx.route == expected_route

    haystack = " ".join(
        (c.get("text", "") + " " + str(c.get("metadata", {}).get("source", ""))).lower()
        for c in ctx.retrieved_chunks
    )
    expected_keywords = expected.get("expected_keywords", [])
    retrieval_hit = (
        True if not expected_keywords or expected_route != "internal_docs"
        else any(kw.lower() in haystack for kw in expected_keywords)
    )

    source_names = " ".join(s.source.lower() for s in ctx.sources)
    expected_sources = expected.get("expected_sources", [])
    source_ok = all(sub.lower() in source_names for sub in expected_sources)

    answer = (ctx.answer or "").lower()
    if expected.get("expect_no_answer"):
        # Retrieving nothing at all also counts: an honest empty hand is the
        # correct behaviour. What must not happen is a confident answer.
        refusal_ok = (not ctx.retrieved_chunks) or any(m in answer for m in REFUSAL_MARKERS)
    else:
        refusal_ok = True

    passed = route_ok and retrieval_hit and source_ok and refusal_ok

    return {
        "passed": passed,
        "route_ok": route_ok,
        "retrieval_hit": retrieval_hit,
        "source_ok": source_ok,
        "refusal_ok": refusal_ok,
        "has_citation": len(ctx.sources) > 0,
        "grounded": ctx.grounded,
        "elapsed": ctx.elapsed,
        "route": ctx.route,
        "chunks": len(ctx.retrieved_chunks),
        "answer_preview": ctx.answer[:120].replace("\n", " "),
    }


# ---------------------------------------------------------------------- #
# Runner
# ---------------------------------------------------------------------- #

def run_evals(pipeline_name: str = "tuk", critic_mode: str = "heuristic",
              enable_expansion: bool = True) -> List[Dict[str, Any]]:
    questions = questions_for(pipeline_name)
    print("=" * 80)
    print(f"Running {pipeline_name} evaluation: {len(questions)} questions")
    print(f"  critic_mode={critic_mode}, enable_expansion={enable_expansion}")
    print("=" * 80)

    pipeline = build_pipeline(pipeline_name, critic_mode, enable_expansion)

    results: List[Dict[str, Any]] = []
    t_total = time.time()

    for i, entry in enumerate(questions, 1):
        question = entry["question"]
        print(f"\n[{i}/{len(questions)}] {question[:70]}")

        ctx = RAGContext(question=question, session_id=f"eval_{pipeline_name}_{i}")
        t0 = time.time()
        try:
            ctx = pipeline.run(ctx)
        except Exception as e:
            print(f"  ✗ EXCEPTION: {e}")
            results.append({
                "question": question,
                "category": entry.get("category", "unknown"),
                "passed": False,
                "route_ok": False,
                "retrieval_hit": False,
                "source_ok": False,
                "refusal_ok": False,
                "has_citation": False,
                "grounded": False,
                "elapsed": time.time() - t0,
                "route": "error",
                "chunks": 0,
                "answer_preview": f"ERROR: {e}",
            })
            continue

        scored = score_question(ctx, entry)
        scored["question"] = question
        scored["category"] = entry.get("category", "unknown")
        results.append(scored)

        status = "✓" if scored["passed"] else "✗"
        print(f"  {status} route={scored['route']} chunks={scored['chunks']} "
              f"grounded={scored['grounded']} cite={scored['has_citation']} "
              f"t={scored['elapsed']:.1f}s")
        if not scored["passed"]:
            why = [
                key for key in ("route_ok", "retrieval_hit", "source_ok", "refusal_ok")
                if not scored[key]
            ]
            print(f"      failed: {', '.join(why)}")
            print(f"      preview: {scored['answer_preview']}")

    total_elapsed = time.time() - t_total

    # ------------------------------------------------------------------ #
    # Report
    # ------------------------------------------------------------------ #

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    n = len(results)
    passed = sum(1 for r in results if r["passed"])
    route_ok = sum(1 for r in results if r["route_ok"])
    retrieval_ok = sum(1 for r in results if r["retrieval_hit"])
    grounded_ok = sum(1 for r in results if r["grounded"])
    cite_ok = sum(1 for r in results if r["has_citation"])
    avg_time = sum(r["elapsed"] for r in results) / n

    print(f"Questions:          {passed}/{n} passed ({100*passed//n}%)")
    print(f"Route accuracy:     {route_ok}/{n} ({100*route_ok//n}%)")
    print(f"Retrieval hits:     {retrieval_ok}/{n} ({100*retrieval_ok//n}%)")
    print(f"Grounded (critic):  {grounded_ok}/{n} ({100*grounded_ok//n}%)")
    print(f"Citation present:   {cite_ok}/{n} ({100*cite_ok//n}%)")
    print(f"Avg latency:        {avg_time:.2f}s")
    print(f"Total time:         {total_elapsed:.1f}s")

    # Category breakdown
    print("\nBy category:")
    categories: Dict[str, Dict[str, int]] = {}
    for r in results:
        cat = r["category"]
        stats = categories.setdefault(cat, {"n": 0, "ok": 0})
        stats["n"] += 1
        if r["passed"]:
            stats["ok"] += 1

    for cat, stats in sorted(categories.items()):
        print(f"  {cat:20s} {stats['ok']}/{stats['n']}")

    failed = [r for r in results if not r["passed"]]
    if failed:
        print("\nFailed questions:")
        for r in failed:
            print(f"  ✗ [{r['category']}] {r['question'][:60]}")
            print(f"      route={r['route']}, chunks={r['chunks']}")
            print(f"      preview: {r['answer_preview']}")

    return results


if __name__ == "__main__":
    run_evals(
        pipeline_name=sys.argv[1] if len(sys.argv) > 1 else "tuk",
        critic_mode=sys.argv[2] if len(sys.argv) > 2 else "heuristic",
        enable_expansion=(sys.argv[3].lower() != "false") if len(sys.argv) > 3 else True,
    )
