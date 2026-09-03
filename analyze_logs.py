"""
Read logs/questions.jsonl and summarise it.

    python analyze_logs.py

This is the payoff for logging STRUCTURED data alongside the readable log.
The text log answers "what happened on that one question?"; this answers
"how is the system doing overall?" - which questions are slow, how often the
grader keeps nothing, how often the retry loop fires and whether it ever helps.
"""

import json
from collections import Counter
from pathlib import Path

JSONL = Path(__file__).parent / "logs" / "questions.jsonl"


def load() -> list[dict]:
    if not JSONL.exists():
        raise SystemExit(f"No log yet at {JSONL}. Ask some questions first.")
    with open(JSONL, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    rows = load()
    print(f"{len(rows)} questions logged\n")

    # --- routing ----------------------------------------------------------
    routes = Counter(r["route"] for r in rows)
    print("ROUTE")
    for route, n in routes.most_common():
        print(f"  {route:8} {n:3d}  ({n / len(rows):.0%})")

    # --- timing -----------------------------------------------------------
    times = sorted(r["total_seconds"] for r in rows)
    print("\nTIME PER QUESTION")
    print(f"  fastest {times[0]:6.1f}s")
    print(f"  median  {times[len(times) // 2]:6.1f}s")
    print(f"  slowest {times[-1]:6.1f}s")

    # --- where the time goes ---------------------------------------------
    per_node: dict[str, list[float]] = {}
    for r in rows:
        for e in r["events"]:
            if "seconds" in e:
                per_node.setdefault(e["node"], []).append(e["seconds"])
    print("\nTIME BY NODE  (total across all questions)")
    for node, secs in sorted(per_node.items(), key=lambda kv: -sum(kv[1])):
        print(f"  {node:15} {sum(secs):7.1f}s  over {len(secs):3d} call(s)"
              f"   avg {sum(secs) / len(secs):5.1f}s")

    # --- the retry loop ---------------------------------------------------
    looped = [r for r in rows if r["attempts"] > 1]
    print(f"\nRETRY LOOP")
    print(f"  fired on {len(looped)}/{len(rows)} questions")
    for r in looped:
        outcome = "gave up" if "don't know" in r["answer"] else "recovered"
        print(f"    {r['attempts']} attempts, {outcome:9}  {r['user_message'][:44]!r}")

    # --- grading ----------------------------------------------------------
    kept_none = sum(1 for r in rows
                    for e in r["events"]
                    if e["node"] == "grade_docs" and e["kept"] == 0)
    grade_calls = sum(1 for r in rows for e in r["events"] if e["node"] == "grade_docs")
    if grade_calls:
        print(f"\nGRADING")
        print(f"  {grade_calls} grading call(s), {kept_none} kept nothing "
              f"({kept_none / grade_calls:.0%})")

    # --- which chunks actually get used ----------------------------------
    used = Counter(c for r in rows for e in r["events"]
                   if e["node"] == "generate" for c in e["chunks_used"])
    if used:
        print("\nMOST-USED CHUNKS")
        for chunk, n in used.most_common(8):
            print(f"  {n:3d}x  {chunk}")

    # --- the slowest questions -------------------------------------------
    print("\nSLOWEST QUESTIONS")
    for r in sorted(rows, key=lambda r: -r["total_seconds"])[:5]:
        print(f"  {r['total_seconds']:6.1f}s  [{r['route']}, {r['attempts']} attempt(s)]"
              f"  {r['user_message'][:44]!r}")


if __name__ == "__main__":
    main()
