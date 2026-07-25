#!/usr/bin/env bash
# Determinism check for the local HF supervisors in run_all_local_content.sh.template.
#
# These no longer need a 10-run CI. Every generative supervisor here now runs greedy:
# HuggingFaceSupervisor pins temperature 0.0, which vLLM takes as `temperature: 0.0`
# and transformers as `do_sample: False`. Unlike the hosted APIs there is no
# server-side model swap and no shared-tenant batching, so a re-run should reproduce
# the original outputs exactly.
#
# "Should" is the point of this script: run ONE extra repetition and diff it against
# the original results. If they match, one run is enough and you can stop here. If they
# don't, the residual is GPU-side (non-deterministic reductions, vLLM continuous
# batching putting different prompts in a batch) -- re-run with N_REPEATS=9 to get a
# real CI over that noise.
#
# Excluded, because judging is a forward pass / prediction head rather than generation:
#   govtech/lionguard-2, and (in run_all_local_jailbreak) piguard, llama-prompt-guard-2.
# saillab/xguard is also excluded: it pins its own temperature (0.01 vllm / 1e-7
# transformers), which is argmax in practice.
#
# Each repetition writes to its own results tree, because Evaluator skips a prompt
# whose result file already exists (see evaluator._load_existing_result).
# Treat your existing run as rep 01; these produce rep 02..(1+N).

set -uo pipefail

SAVE_DIR="content-moderation-leaderboard/results"   # the original, rep 01
REPEAT_ROOT="content-moderation-leaderboard/results_repeats"
N_REPEATS="${N_REPEATS:-1}"     # 1 = determinism check; set to 9 for a full CI sweep
START_REP="${START_REP:-2}"

CONFIG="configs/content_moderation.json"

source .venv/bin/activate

for i in $(seq "$START_REP" $((START_REP + N_REPEATS - 1))); do
    REP_DIR="$REPEAT_ROOT/rep_$(printf '%02d' "$i")"
    echo "=================== repetition $i -> $REP_DIR ==================="

    # Supervisors that don't require used_for (both datasets via config)
    python run_eval.py --model-id="google/shieldgemma-2b" --type="hf" --save_dir=$REP_DIR --config=$CONFIG
    python run_eval.py --model-id="google/shieldgemma-27b" --type="hf" --save_dir=$REP_DIR --config=$CONFIG
    python run_eval.py --model-id="qwen/qwen3guard-gen-8b" --type="hf" --save_dir=$REP_DIR --config=$CONFIG
    python run_eval.py --model-id="qwen/qwen3guard-gen-0.6b" --type="hf" --save_dir=$REP_DIR --config=$CONFIG
    python run_eval.py --model-id="ibm-granite/granite-guardian-3.3-8b" --type="hf" --save_dir=$REP_DIR --config=$CONFIG

    # Supervisors that require used_for (per-dataset)
    for USED_FOR in input output; do
        # NOTE: gpt-oss-safeguard-120b needs an H200. Comment it out on <90GB VRAM.
        python run_eval.py --model-id="openai/gpt-oss-safeguard-120b" --supervisor-kwarg used_for=$USED_FOR --type="hf" --save_dir=$REP_DIR --dataset-id="bells-o-project/content-moderation-$USED_FOR" --usage="content_moderation" --truthy-value="!Benign"
        python run_eval.py --model-id="rakancorle1/thinkguard" --type="hf" --supervisor-kwarg used_for=$USED_FOR --save_dir=$REP_DIR --dataset-id="bells-o-project/content-moderation-$USED_FOR" --usage="content_moderation" --truthy-value="!Benign"
        python run_eval.py --model-id="allenai/wildguard" --type="hf" --supervisor-kwarg used_for=$USED_FOR --save_dir=$REP_DIR --dataset-id="bells-o-project/content-moderation-$USED_FOR" --usage="content_moderation" --truthy-value="!Benign"
        python run_eval.py --model-id="toxicityprompts/polyguard-ministral" --type="hf" --supervisor-kwarg used_for=$USED_FOR --save_dir=$REP_DIR --dataset-id="bells-o-project/content-moderation-$USED_FOR" --usage="content_moderation" --truthy-value="!Benign"
        python run_eval.py --model-id="toxicityprompts/polyguard-qwen" --type="hf" --supervisor-kwarg used_for=$USED_FOR --save_dir=$REP_DIR --dataset-id="bells-o-project/content-moderation-$USED_FOR" --usage="content_moderation" --truthy-value="!Benign"
        python run_eval.py --model-id="nvidia/llama-3.1-nemotron-safety-guard-8b-v3" --type="hf" --supervisor-kwarg used_for=$USED_FOR --save_dir=$REP_DIR --dataset-id="bells-o-project/content-moderation-$USED_FOR" --usage="content_moderation" --truthy-value="!Benign"
    done
done

FIRST_REP="$REPEAT_ROOT/rep_$(printf '%02d' "$START_REP")"
echo
echo "Done. Check determinism against the original run with:"
echo "  python scripts/diff_reps.py $SAVE_DIR $FIRST_REP"
echo
echo "If outputs diverge, re-run with N_REPEATS=9 and aggregate:"
echo "  N_REPEATS=9 START_REP=3 bash run_repeats_local_content.sh"
echo "  python scripts/aggregate_repeats.py --leaderboard ../content-moderation-leaderboard \\"
echo "      --rep-dir $SAVE_DIR --rep-glob '$REPEAT_ROOT/rep_*'"
