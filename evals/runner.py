"""
Evaluation harness for the cancer pipeline.
...
"""

import io
import time
import sys
from typing import Dict, Any, List

# Force UTF-8 stdout/stderr on Windows so redirected output doesn't
# crash on ✓/✗ characters. Also silences the copy-paste mojibake.
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from app.rag import RAGContext
from app.rag.configs.cancer_pipeline import build_cancer_pipeline

from evals.questions import EVAL_QUESTIONS


# ---------------------------------------------------------------------- #
# Scoring
# ---------------------------------------------------------------------- #

def score_question(ctx: RAGContext, expected: Dict[str, Any]) -> Dict[str, Any]:
    """Score a single pipeline run against its expected outcome."""
    expected_route = expected.get("expected_route", "internal_docs")
    route_ok = ctx.route == expected_route

    # Retrieval hit: any expected keyword appears in any retrieved chunk?
    expected_keywords = expected.get("expected_keywords", [])
    if not expected_keywords or expected_route != "internal_docs":
        retrieval_hit = True  # not applicable
    else:
        combined = " ".join(
            (c.get("text", "") + " " + c.get("metadata", {}).get("title", "")).lower()
            for c in ctx.retrieved_chunks
        )
        retrieval_hit = any(kw.lower() in combined for kw in expected_keywords)

    # Citation presence: at least one source in the final context?
    has_citation = len(ctx.sources) > 0

    # Grounded flag (critic verdict)
    grounded = ctx.grounded

    return {
        "route_ok": route_ok,
        "retrieval_hit": retrieval_hit,
        "has_citation": has_citation,
        "grounded": grounded,
        "elapsed": ctx.elapsed,
        "route": ctx.route,
        "chunks": len(ctx.retrieved_chunks),
        "answer_preview": ctx.answer[:120].replace("\n", " "),
    }


# ---------------------------------------------------------------------- #
# Runner
# ---------------------------------------------------------------------- #

def run_evals(critic_mode: str = "heuristic", enable_expansion: bool = True):
    print("=" * 80)
    print(f"Running evaluation: {len(EVAL_QUESTIONS)} questions")
    print(f"  critic_mode={critic_mode}, enable_expansion={enable_expansion}")
    print("=" * 80)

    pipeline = build_cancer_pipeline(
        critic_mode=critic_mode,
        enable_expansion=enable_expansion,
    )

    results: List[Dict[str, Any]] = []
    t_total = time.time()

    for i, entry in enumerate(EVAL_QUESTIONS, 1):
        question = entry["question"]
        print(f"\n[{i}/{len(EVAL_QUESTIONS)}] {question[:70]}")

        ctx = RAGContext(question=question, session_id=f"eval_{i}")
        t0 = time.time()
        try:
            ctx = pipeline.run(ctx)
        except Exception as e:
            print(f"  ✗ EXCEPTION: {e}")
            results.append({
                "question": question,
                "category": entry.get("category", "unknown"),
                "route_ok": False,
                "retrieval_hit": False,
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

        status = "✓" if (scored["route_ok"] and scored["retrieval_hit"]) else "✗"
        print(f"  {status} route={scored['route']} chunks={scored['chunks']} "
              f"grounded={scored['grounded']} cite={scored['has_citation']} "
              f"t={scored['elapsed']:.1f}s")

    total_elapsed = time.time() - t_total

    # ------------------------------------------------------------------ #
    # Report
    # ------------------------------------------------------------------ #

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    n = len(results)
    route_ok = sum(1 for r in results if r["route_ok"])
    retrieval_ok = sum(1 for r in results if r["retrieval_hit"])
    grounded_ok = sum(1 for r in results if r["grounded"])
    cite_ok = sum(1 for r in results if r["has_citation"])
    avg_time = sum(r["elapsed"] for r in results) / n

    print(f"Questions:          {n}")
    print(f"Route accuracy:     {route_ok}/{n} ({100*route_ok//n}%)")
    print(f"Retrieval hits:     {retrieval_ok}/{n} ({100*retrieval_ok//n}%)")
    print(f"Grounded (critic):  {grounded_ok}/{n} ({100*grounded_ok//n}%)")
    print(f"Citation present:   {cite_ok}/{n} ({100*cite_ok//n}%)")
    print(f"Avg latency:        {avg_time:.2f}s")
    print(f"Total time:         {total_elapsed:.1f}s")

    # Category breakdown
    print("\nBy category:")
    categories = {}
    for r in results:
        cat = r["category"]
        categories.setdefault(cat, {"n": 0, "ok": 0})
        categories[cat]["n"] += 1
        if r["route_ok"] and r["retrieval_hit"]:
            categories[cat]["ok"] += 1

    for cat, stats in sorted(categories.items()):
        print(f"  {cat:20s} {stats['ok']}/{stats['n']}")

    # Failed questions
    failed = [r for r in results if not (r["route_ok"] and r["retrieval_hit"])]
    if failed:
        print("\nFailed questions:")
        for r in failed:
            print(f"  ✗ [{r['category']}] {r['question'][:60]}")
            print(f"      route={r['route']}, chunks={r['chunks']}")
            print(f"      preview: {r['answer_preview']}")

    return results


if __name__ == "__main__":
    critic_mode = sys.argv[1] if len(sys.argv) > 1 else "llm"
    run_evals(critic_mode=critic_mode)