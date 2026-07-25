#!/usr/bin/env bash
# Re-run the REST supervisors from run_all_rest_content.sh.template that are STILL
# non-deterministic, N extra times, so their scores get a CI.
#
# Only two models qualify. gpt-5-nano and gpt-5.2 are reasoning models: the OpenAI
# API rejects temperature != 1 on them, so openai_mapper deliberately skips the
# temperature field (see the "gpt-5" in model_name guard) and they sample exactly
# as they did before the temperature pin.
#
# Everything else in run_all_rest_content.sh.template is a single-run model now:
#   - deterministic classifiers / policy engines, unaffected by the pin:
#       azure-analyze-text, lakeraguard-default, openai-moderation, bedrock-guardrail
#   - LLM judges now pinned to temperature 0.0 via RestSupervisor.temperature:
#       claude-sonnet-4-6, claude-haiku-4-5, gemini-2.5-flash, ministral-3b-2512,
#       mistral-large-3, grok-4-1-fast-non-reasoning, gpt-oss-20b, gpt-oss-120b,
#       gpt-oss-safeguard-20b, virtueguard-text-lite, llama-guard-4-12b
# Run those with run_all_rest_content.sh.template as usual.
#
# NOTE: temperature 0 is greedy decoding, not a determinism guarantee — batching and
# MoE-routing nondeterminism still move outputs, and providers update models behind
# version aliases. To check rather than assume, run the pinned set twice and diff the
# per-prompt results instead of paying for ten passes.
#
# Each repetition writes to its own results tree, because Evaluator skips a prompt
# whose result file already exists (see evaluator._load_existing_result).
# Treat your existing run as rep 01; these produce rep 02..(1+N).

set -uo pipefail

SAVE_DIR="leaderboard/results"          # the original, rep 01
REPEAT_ROOT="leaderboard/results_repeats"
N_REPEATS="${N_REPEATS:-9}"
START_REP="${START_REP:-2}"

BATCH_SIZE=40

source .venv/bin/activate

for i in $(seq "$START_REP" $((START_REP + N_REPEATS - 1))); do
    REP_DIR="$REPEAT_ROOT/rep_$(printf '%02d' "$i")"
    echo "=================== repetition $i -> $REP_DIR ==================="

    # Reasoning models — temperature is not settable, so these still sample
    for USED_FOR in input output; do
        python run_eval.py --model-id="openai-classification" --type="rest" --supervisor-kwarg used_for=$USED_FOR --save_dir=$REP_DIR --lab="openai" --model_name="gpt-5-nano" --dataset-id="bells-o-project/content-moderation-$USED_FOR" --usage="content_moderation" --truthy-value="!Benign" --batch_size=$BATCH_SIZE
        python run_eval.py --model-id="openai-classification" --type="rest" --supervisor-kwarg model=gpt-5.2-2025-12-11 --supervisor-kwarg used_for=$USED_FOR --save_dir=$REP_DIR --lab="openai" --model_name="gpt-5.2" --dataset-id="bells-o-project/content-moderation-$USED_FOR" --usage="content_moderation" --truthy-value="!Benign" --batch_size=$BATCH_SIZE
    done
done

echo "Done. Aggregate with:"
echo "  python scripts/aggregate_repeats.py --leaderboard ../content-moderation-leaderboard --rep-dir $SAVE_DIR --rep-glob '$REPEAT_ROOT/rep_*'"
