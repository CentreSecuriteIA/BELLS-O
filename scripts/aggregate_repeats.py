#!/usr/bin/env python3
"""Aggregate metrics across repeated evaluation runs into mean +/- 95% CI.

Each repetition lives in its own results tree (see run_repeats_*.sh), because the
Evaluator skips prompts whose result file already exists. This script computes the
leaderboard metrics for every repetition and then, per
(provider / dataset / model), reports mean, sample std, and a 95% confidence
interval on the mean using the t distribution.

Usage:
    python scripts/aggregate_repeats.py \\
        --leaderboard jailbreak-leaderboard \\
        --rep-dir leaderboard/results \\
        --rep-glob 'leaderboard/results_repeats/rep_*' \\
        --output ci_results.json

`--rep-dir` may be repeated and `--rep-glob` expanded; both are pooled, so pass
the original run as one of them if it should count as repetition 1.
"""

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


# Metrics that are numeric and worth putting a CI on.
METRIC_KEYS = ("accuracy", "fpr", "mean_latency")

# 95% two-sided t critical values for n-1 degrees of freedom, index = n.
_T95 = {
    2: 12.706,
    3: 4.303,
    4: 3.182,
    5: 2.776,
    6: 2.571,
    7: 2.447,
    8: 2.365,
    9: 2.306,
    10: 2.262,
    11: 2.228,
    12: 2.201,
    13: 2.179,
    14: 2.160,
    15: 2.145,
    16: 2.131,
    17: 2.120,
    18: 2.110,
    19: 2.101,
    20: 2.093,
}


def _t_critical(n: int) -> float:
    """Return the two-sided 95% t critical value for a sample of size n."""
    if n < 2:
        return float("nan")
    if n in _T95:
        return _T95[n]
    return 1.96  # large-n normal approximation


def _load_leaderboard(leaderboard_dir: Path):
    """Import Metrics and the combination finder from a leaderboard directory."""
    sys.path.insert(0, str(leaderboard_dir.resolve()))
    from compute_metrics import Metrics  # type: ignore[import-not-found]
    from run_compute_metrics import find_all_result_combinations  # type: ignore[import-not-found]

    return Metrics, find_all_result_combinations


def _metrics_for_rep(
    rep_dir: Path,
    mapping_file: Path,
    metrics_cls,
    find_combinations,
) -> dict[str, dict[str, Any]]:
    """Compute all metrics for one repetition, keyed by provider/dataset/model."""
    calculator = metrics_cls(rep_dir, mapping_file=mapping_file)
    out: dict[str, dict[str, Any]] = {}

    for provider, dataset, model in find_combinations(rep_dir):
        metrics = calculator.compute_all_metrics(
            model_provider_use_case=provider, dataset_name=dataset, model_name=model
        )
        out[f"{provider}/{dataset}/{model}"] = metrics

    return out


def _summarize(values: list[float]) -> dict[str, Any]:
    """Return mean, std, n and a 95% CI on the mean for a list of observations."""
    n = len(values)
    mean = statistics.fmean(values)
    if n < 2:
        return {"mean": mean, "std_dev": 0.0, "n": n, "ci_95_lower": mean, "ci_95_upper": mean, "half_width": 0.0}

    std = statistics.stdev(values)
    half_width = _t_critical(n) * std / (n**0.5)
    return {
        "mean": mean,
        "std_dev": std,
        "n": n,
        "ci_95_lower": mean - half_width,
        "ci_95_upper": mean + half_width,
        "half_width": half_width,
    }


def aggregate(rep_dirs: list[Path], leaderboard_dir: Path, mapping_file: Path) -> dict[str, Any]:
    """Compute per-model mean and 95% CI across all repetitions."""
    metrics_cls, find_combinations = _load_leaderboard(leaderboard_dir)

    # observations[key][metric] = [value per repetition]
    observations: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    seen_in: dict[str, list[str]] = defaultdict(list)

    for rep_dir in rep_dirs:
        print(f"--- computing metrics for {rep_dir}")
        rep_metrics = _metrics_for_rep(rep_dir, mapping_file, metrics_cls, find_combinations)
        print(f"    {len(rep_metrics)} model/dataset combinations")

        for key, metrics in rep_metrics.items():
            seen_in[key].append(str(rep_dir))
            for metric in METRIC_KEYS:
                value = metrics.get(metric)
                if isinstance(value, (int, float)):  # "N/A" is returned when undefined
                    observations[key][metric].append(float(value))

    summary: dict[str, Any] = {}
    for key in sorted(observations):
        provider, dataset, model = key.split("/", 2)
        entry: dict[str, Any] = {
            "model_provider_use_case": provider,
            "dataset_name": dataset,
            "model_name": model,
            "n_repetitions": len(seen_in[key]),
            "repetitions": seen_in[key],
        }
        for metric, values in observations[key].items():
            entry[metric] = _summarize(values)
        summary[key] = entry

    return summary


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--leaderboard",
        type=Path,
        required=True,
        help="Leaderboard directory holding compute_metrics.py (e.g. jailbreak-leaderboard)",
    )
    parser.add_argument(
        "--rep-dir",
        type=Path,
        action="append",
        default=[],
        help="A results tree for one repetition (repeatable)",
    )
    parser.add_argument(
        "--rep-glob",
        type=str,
        action="append",
        default=[],
        help="Glob matching repetition results trees, e.g. 'leaderboard/results_repeats/rep_*' (repeatable)",
    )
    parser.add_argument(
        "--mapping-file",
        type=Path,
        help="model_info_mapping.json to use (defaults to <leaderboard>/model_info_mapping.json)",
    )
    parser.add_argument("--output", type=Path, help="Write the aggregated summary here as JSON")
    args = parser.parse_args()

    rep_dirs: list[Path] = list(args.rep_dir)
    for pattern in args.rep_glob:
        rep_dirs.extend(sorted(p for p in Path().glob(pattern) if p.is_dir()))

    rep_dirs = [d for d in rep_dirs if d.is_dir()]
    if not rep_dirs:
        parser.error("No repetition directories found. Pass --rep-dir and/or --rep-glob.")

    mapping_file = args.mapping_file or (args.leaderboard / "model_info_mapping.json")

    print(f"Aggregating {len(rep_dirs)} repetitions using {args.leaderboard}\n")
    summary = aggregate(rep_dirs, args.leaderboard, mapping_file)

    print(f"\n{'model':<60} {'n':>3} {'accuracy mean':>14} {'95% CI':>24}")
    print("-" * 105)
    for key, entry in summary.items():
        acc = entry.get("accuracy")
        if not acc:
            continue
        ci = f"[{acc['ci_95_lower']:.4f}, {acc['ci_95_upper']:.4f}]"
        print(f"{key[:60]:<60} {entry['n_repetitions']:>3} {acc['mean']:>14.4f} {ci:>24}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
