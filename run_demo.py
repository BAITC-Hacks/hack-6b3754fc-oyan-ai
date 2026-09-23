"""Run saved requests through the real application core, without a browser."""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

from data_loader import load_contractors
from recommender import recommend

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", help="Run one case by name; default: all cases")
    parser.add_argument("--verify", action="store_true", help="Compare responses with saved real results")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "demo_queries.json").read_text(encoding="utf-8"))
    contractors = load_contractors(ROOT / manifest["dataset_path"])
    cases = [case for case in manifest["cases"] if args.case is None or case["name"] == args.case]
    if not cases:
        parser.error("Unknown case name")
    for case in cases:
        started = perf_counter()
        result = recommend(case["query"], contractors)
        elapsed = perf_counter() - started
        if args.verify and result != case["response"]:
            raise SystemExit(f"FAIL: {case['name']}: response differs from saved result")
        print(json.dumps({"case": case["name"], "elapsed_seconds": round(elapsed, 6),
                          "response": result}, ensure_ascii=False, indent=2))
    if args.verify:
        print(f"OK: {len(cases)} saved responses match the real core")


if __name__ == "__main__":
    # Preserve Cyrillic and the tenge sign in Windows terminals and captured pipes.
    sys.stdout.reconfigure(encoding="utf-8")
    main()
