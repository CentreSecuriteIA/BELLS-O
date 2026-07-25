#!/usr/bin/env python3
"""Compare two evaluation results trees prompt-by-prompt.

Used to check whether a re-run reproduces the original outputs — the cheap alternative
to a 10-run CI for supervisors that should now be deterministic (see run_repeats_local_content.sh).

Only the model's verdict is compared. `metadata` (latency, token counts, timestamps) differs
on every run by construction and is ignored.

Two levels of divergence are reported separately:
  * result  — the mapped `output_result`, i.e. what the leaderboard scores. This is what matters.
  * raw     — the verbatim `output_raw`. Raw drift with a stable result means the wording moved
              but the verdict held; worth knowing, but it does not move any metric.

Usage:
    python scripts/diff_reps.py <results_dir_a> <results_dir_b>

Exits 0 when every compared verdict matches, 1 otherwise.
"""

import json
import sys
from pathlib import Path
from typing import Any


def _load_tree(root: Path) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    """Load every result file under a results tree, keyed by provider/dataset/model/prompt."""
    out: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for path in root.glob("*/*/*/*.json"):
        provider, dataset, model, prompt = (
            path.parts[-4],
            path.parts[-3],
            path.parts[-2],
            path.stem,
        )
        try:
            out[(provider, dataset, model, prompt)] = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: could not read {path}: {e}", file=sys.stderr)
    return out


def main() -> int:
    """Entry point."""
    if len(sys.argv) != 3:
        print(__doc__)
        return 2

    root_a, root_b = Path(sys.argv[1]), Path(sys.argv[2])
    for root in (root_a, root_b):
        if not root.is_dir():
            print(f"Not a directory: {root}", file=sys.stderr)
            return 2

    a, b = _load_tree(root_a), _load_tree(root_b)
    shared = sorted(a.keys() & b.keys())

    if not shared:
        print(f"No overlapping prompts between {root_a} and {root_b}.")
        print("Check that both trees use the same run_id (--model_name) layout.")
        return 2

    # per (provider, dataset, model): [compared, result_diffs, raw_only_diffs]
    stats: dict[tuple[str, str, str], list[int]] = {}
    examples: list[str] = []

    for key in shared:
        group = key[:3]
        row = stats.setdefault(group, [0, 0, 0])
        row[0] += 1
        ra, rb = a[key].get("output_result"), b[key].get("output_result")
        if ra != rb:
            row[1] += 1
            if len(examples) < 10:
                examples.append(f"  {'/'.join(group)}  prompt {key[3][:24]}…  {ra} -> {rb}")
        elif a[key].get("output_raw") != b[key].get("output_raw"):
            row[2] += 1

    only_a, only_b = len(a.keys() - b.keys()), len(b.keys() - a.keys())

    print(f"Comparing {root_a}  vs  {root_b}")
    print(f"{len(shared)} prompts in both" + (f"  ({only_a} only in A, {only_b} only in B)" if only_a or only_b else ""))
    print()
    print(f"{'provider/dataset/model':<70} {'n':>6} {'result≠':>8} {'raw≠':>6}")
    print("-" * 94)
    for group in sorted(stats):
        n, res_diff, raw_diff = stats[group]
        label = "/".join(group)
        flag = "  <-- diverged" if res_diff else ""
        print(f"{label[:70]:<70} {n:>6} {res_diff:>8} {raw_diff:>6}{flag}")

    total_res = sum(v[1] for v in stats.values())
    total_raw = sum(v[2] for v in stats.values())
    print()
    if total_res:
        print(f"{total_res}/{len(shared)} verdicts differ — these runs are NOT reproducible.")
        print("Examples:")
        print("\n".join(examples))
        print("\nRe-run with N_REPEATS=9 and use scripts/aggregate_repeats.py for a CI.")
    else:
        print(f"All {len(shared)} verdicts identical.", end=" ")
        if total_raw:
            print(f"({total_raw} differ in raw text only — verdicts unaffected.)")
        else:
            print("Raw outputs identical too.")
        print("One run is sufficient for these supervisors.")

    return 1 if total_res else 0


if __name__ == "__main__":
    sys.exit(main())
