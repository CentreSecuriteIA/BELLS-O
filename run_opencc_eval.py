"""Run the BELLS-O Evaluator against a locally-served OpenCC supervisor.

Reports detection rate, FPR, per-category / per-technique breakdowns, **per-sample latency**,
and emits a BELLS-O leaderboard row (with cost). Saves per-sample rows (JSONL) and a summary
(JSON).

Usage:
    # content moderation (serve OpenCC with config.cm-only.yaml)
    python run_opencc_eval.py --usage content_moderation --limit 0

    # jailbreak detection (serve OpenCC with config.jb-only.yaml)
    python run_opencc_eval.py --usage jailbreak --limit 0 \
        --dataset centrepourlasecuriteia/jailbreak-dataset --positive-when always
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from bells_o import Evaluator, HuggingFaceDataset, Result, Usage
from bells_o.evaluator import DatasetConfig, SupervisorConfig
from bells_o.supervisors.rest import OpenCCSupervisor

BENIGN_LABEL = "Benign"
LEADERBOARD_HEADER = (
    "Rank\tModel Snapshot\tModel Developer\tProvider\tModel Type\tDetection Rate (%)\t"
    "FPR (%)\tLatency CI 95% (ms)\tMean Latency (ms)\tCompute Access\tTotal Cost\t"
    "Cost per 1M units\tCost per h\tCost Additional Info\tExecution Info"
)


def target_map_fn_for(usage_type, positive_when):
    def fn(value: str) -> Result:
        flagged = True if positive_when == "always" else (value != BENIGN_LABEL)
        return Result(**{usage_type: flagged})
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--usage", default="content_moderation", choices=["content_moderation", "jailbreak"])
    ap.add_argument("--dataset", default="bells-o-project/content-moderation-input")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--target-col", default="category")
    ap.add_argument("--positive-when", default="not-benign", choices=["not-benign", "always"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out", default=None, help="summary JSON path")
    ap.add_argument("--rows-out", default=None, help="per-sample JSONL path (flat, optional)")
    ap.add_argument("--save-dir", default="results",
                    help="BELLS-O native per-sample log dir (one JSON per prompt, incl. latency)")
    # leaderboard / cost
    ap.add_argument("--cost-per-hour", type=float, default=2.39, help="GPU $/hour (RunPod H100)")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-0.8B", help="tokenizer for input-token count")
    ap.add_argument("--model-snapshot", default=None)
    ap.add_argument("--developer", default="CeSIA")
    ap.add_argument("--provider", default="RunPod")
    ap.add_argument("--model-type", default="Specialized")
    ap.add_argument("--execution-info", default=None)
    args = ap.parse_args()

    usage = Usage(args.usage)
    dataset_conf = DatasetConfig(
        type=HuggingFaceDataset,
        kwargs={"name": args.dataset, "usage": usage,
                "target_map_fn": target_map_fn_for(args.usage, args.positive_when),
                "input_column": "prompt"},
        input_column="prompt", target_column=args.target_col,
    )
    supervisor_conf = SupervisorConfig(type=OpenCCSupervisor, kwargs={"host": args.host, "port": args.port})
    evaluator = Evaluator(dataset_conf, supervisor_conf, save_dir=args.save_dir, verbose=True, batch_size=1)

    n = len(evaluator.dataset)
    if args.limit and args.limit < n:
        step = max(1, n // args.limit)
        indices = list(range(0, n, step))[: args.limit]
    else:
        indices = list(range(n))
    print(f"Dataset {args.dataset}: {n} samples | evaluating {len(indices)} | usage={args.usage}")

    # warm up the server so the first timed sample isn't the model-load
    evaluator.supervisor("warmup")

    run_id = f"opencc-{args.usage}"
    t0 = time.time()
    evaluator.run(indices=indices, run_id=run_id, save=True, verbose=True)
    wall_s = time.time() - t0
    saved_dir = Path(args.save_dir) / evaluator.dataset.clean_name / run_id

    id2sample = {s["prompt_id"]: s for s in evaluator.dataset}
    rows = []
    for r in evaluator.runs[run_id].values():
        s = id2sample.get(r["metadata"]["prompt_id"], {})
        rows.append({
            "prompt_id": r["metadata"]["prompt_id"],
            "pred": bool(r["output_result"][args.usage]),
            "target": bool(r["target_result"][args.usage]),
            "category": s.get("category"),
            "technique_type": s.get("technique_type"),
            "latency_ms": float(r["metadata"].get("latency", float("nan"))) * 1000.0,
            "prompt": s.get("prompt", ""),
        })

    # ---- confusion / detection / fpr ----
    tp = sum(x["target"] and x["pred"] for x in rows)
    fn = sum(x["target"] and not x["pred"] for x in rows)
    fp = sum(not x["target"] and x["pred"] for x in rows)
    tn = sum(not x["target"] and not x["pred"] for x in rows)
    total = tp + tn + fp + fn
    has_neg = (fp + tn) > 0
    det = tp / (tp + fn) if (tp + fn) else float("nan")
    fpr = fp / (fp + tn) if has_neg else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    acc = (tp + tn) / total if total else 0.0

    # ---- latency ----
    lat = np.array([x["latency_ms"] for x in rows if x["latency_ms"] == x["latency_ms"]])
    mean_lat = float(lat.mean())
    sem = float(lat.std(ddof=1) / np.sqrt(len(lat))) if len(lat) > 1 else 0.0
    ci_lo, ci_hi = mean_lat - 1.96 * sem, mean_lat + 1.96 * sem
    p50, p95 = float(np.percentile(lat, 50)), float(np.percentile(lat, 95))

    # ---- input tokens (OpenCC returns 0; count locally with the model tokenizer) ----
    total_input_tokens = None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        total_input_tokens = int(sum(len(tok(x["prompt"]).input_ids) for x in rows))
    except Exception as e:
        print(f"WARN: could not count input tokens ({type(e).__name__}: {e})")

    # ---- cost ----
    total_cost = wall_s / 3600.0 * args.cost_per_hour
    cost_per_1m_in = (total_cost * 1_000_000 / total_input_tokens) if total_input_tokens else None

    print("\n================ OVERALL ================")
    print(f"usage={args.usage}  n={total}  (positive={tp+fn}, negative={tn+fp})")
    print(f"  TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"  detection_rate={det:.4f}   FPR={'%.4f' % fpr if has_neg else 'n/a (no negatives here)'}")
    print(f"  accuracy={acc:.4f}  precision={prec:.4f}")
    print(f"  latency_ms: mean={mean_lat:.1f}  CI95=[{ci_lo:.1f},{ci_hi:.1f}]  p50={p50:.1f}  p95={p95:.1f}")
    print(f"  wall={wall_s:.1f}s  input_tokens={total_input_tokens}  total_cost=${total_cost:.4f}  "
          f"cost/1M_in={'$%.4f' % cost_per_1m_in if cost_per_1m_in else 'n/a'}")

    def breakdown(key, title):
        if not any(x[key] is not None for x in rows):
            return None
        agg = defaultdict(lambda: {"n": 0, "flagged": 0})
        for x in rows:
            a = agg[x[key]]; a["n"] += 1; a["flagged"] += int(x["pred"])
        print(f"\n=========== BY {title} (flag rate) ===========")
        for k in sorted(agg, key=lambda z: (z != BENIGN_LABEL, str(z))):
            a = agg[k]; rate = a["flagged"] / a["n"]
            tag = " (FPR)" if (k == BENIGN_LABEL and has_neg) else ""
            print(f"  {str(k):30s} {a['n']:>5d} {a['flagged']:>8d} {rate:>7.3f}{tag}")
        return {str(k): {"n": v["n"], "flagged": v["flagged"], "rate": v["flagged"] / v["n"]}
                for k, v in agg.items()}

    by_cat = breakdown("category", "CATEGORY")
    by_tech = breakdown("technique_type", "TECHNIQUE_TYPE")

    # ---- leaderboard row ----
    snapshot = args.model_snapshot or (
        "opencc-cm-escalation" if args.usage == "content_moderation" else "opencc-jb-escalation")
    exec_info = args.execution_info or (
        "This model was run on an H100 NVL (95GB) on RunPod via OpenCC's local FastAPI "
        "/check endpoint (hf_classifier backend, batch_size=1).")
    cost_note = ("Output token cost is disregarded (local classifier, no generated tokens). "
                 "The cost per 1M input tokens is estimated as total_cost * (1,000,000 / total_input_tokens).")
    cost_per_1m_units = (f"Input: ${cost_per_1m_in:.4f}/1M, Output: $0.0/1M"
                         if cost_per_1m_in else "Input: n/a, Output: $0.0/1M")
    fpr_cell = f"{fpr*100:.2f}%" if has_neg else "n/a"
    row = "\t".join([
        "", snapshot, args.developer, args.provider, args.model_type,
        f"{det*100:.2f}%", fpr_cell,
        f"[{ci_lo:.0f}, {ci_hi:.0f}]", f"{mean_lat:.0f}",
        "Local", f"${total_cost:.2f}", cost_per_1m_units, f"${args.cost_per_hour:.2f}",
        cost_note, exec_info,
    ])
    print("\n================ LEADERBOARD ROW ================")
    print(LEADERBOARD_HEADER)
    print(row)

    out = Path(args.out) if args.out else Path(f"results_{args.usage}.json")
    out.write_text(json.dumps({
        "usage": args.usage, "dataset": args.dataset, "n": total,
        "overall": {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "detection_rate": det,
                    "fpr": (fpr if has_neg else None), "accuracy": acc, "precision": prec},
        "latency_ms": {"mean": mean_lat, "ci95": [ci_lo, ci_hi], "p50": p50, "p95": p95},
        "cost": {"wall_s": wall_s, "cost_per_hour": args.cost_per_hour, "total_cost": total_cost,
                 "total_input_tokens": total_input_tokens, "cost_per_1m_input": cost_per_1m_in},
        "by_category": by_cat, "by_technique_type": by_tech,
        "leaderboard_header": LEADERBOARD_HEADER, "leaderboard_row": row,
    }, indent=2))
    if args.rows_out:
        with open(args.rows_out, "w") as f:
            for x in rows:
                f.write(json.dumps({k: x[k] for k in
                        ("prompt_id", "pred", "target", "category", "technique_type", "latency_ms")}) + "\n")
        print(f"Saved per-sample rows -> {Path(args.rows_out).resolve()}")
    print(f"Saved summary -> {out.resolve()}")
    print(f"Per-sample BELLS-O logs (one JSON/prompt, incl. latency) -> {saved_dir.resolve()}")


if __name__ == "__main__":
    main()
