SAVE_DIR="leaderboard/results"
source .venv/bin/activate

# all HF supervisors for the benchmark
python run_eval.py "saillab/xguard" --type="hf" --save_dir=$SAVE_DIR
python run_eval.py "saillab/xguard" --type="hf" --save_dir=$SAVE_DIR
python run_eval.py "saillab/xguard" --type="hf" --save_dir=$SAVE_DIR
python run_eval.py "saillab/xguard" --type="hf" --save_dir=$SAVE_DIR
python run_eval.py "openai/gptossafeguard-20b" --type="hf" --save_dir=$SAVE_DIR
python run_eval.py "openai/gptossafeguard-20b" --type="hf" --save_dir=$SAVE_DIR
python run_eval.py "openai/gptossafeguard-20b" --type="hf" --save_dir=$SAVE_DIR
python run_eval.py "openai/gptossafeguard-20b" --type="hf" --save_dir=$SAVE_DIR
python run_eval.py "openai/gptossafeguard-120b" --type="hf" --save_dir=$SAVE_DIR
python run_eval.py "google/shieldgemma-2b" --type="hf" --save_dir=$SAVE_DIR
python run_eval.py "google/shieldgemma-9b" --type="hf" --save_dir=$SAVE_DIR
python run_eval.py "google/shieldgemma-27b" --type="hf" --save_dir=$SAVE_DIR
python run_eval.py "qwen/qwen3guard-gen-8b" --type="hf" --save_dir=$SAVE_DIR
python run_eval.py "qwen/qwen3guard-gen-4b" --type="hf" --save_dir=$SAVE_DIR
python run_eval.py "qwen/qwen3guard-gen-0.6b" --type="hf" --save_dir=$SAVE_DIR
python run_eval.py "rakancorle1/thinkguard" --type="hf" --save_dir=$SAVE_DIR
python run_eval.py "allenai/wildguard" --type="hf" --save_dir=$SAVE_DIR
python run_eval.py "toxicityprompts/polyguard-ministral" --type="hf" --save_dir=$SAVE_DIR
python run_eval.py "toxicityprompts/polyguard-qwen" --type="hf" --save_dir=$SAVE_DIR
python run_eval.py "toxicityprompts/polyguard-qwen-smol" --type="hf" --save_dir=$SAVE_DIR
python run_eval.py "ibm-granite/granite-guardian-3.0-2b" --type="hf" --save_dir=$SAVE_DIR
python run_eval.py "ibm-granite/granite-guardian-3.0-8b" --type="hf" --save_dir=$SAVE_DIR
python run_eval.py "ibm-granite/granite-guardian-3.1-2b" --type="hf" --save_dir=$SAVE_DIR
python run_eval.py "ibm-granite/granite-guardian-3.1-8b" --type="hf" --save_dir=$SAVE_DIR
python run_eval.py "ibm-granite/granite-guardian-3.2-5b" --type="hf" --save_dir=$SAVE_DIR
python run_eval.py "ibm-granite/granite-guardian-3.2-3b-a800m" --type="hf" --save_dir=$SAVE_DIR
python run_eval.py "ibm-granite/granite-guardian-3.3-8b" --type="hf" --save_dir=$SAVE_DIR
python run_eval.py "govtech/lionguard-2" --type="hf" --kwargs='{"backend":"transformers"}'--save_dir=$SAVE_DIR
python run_eval.py "govtech/lionguard-2.1" --type="hf" --kwargs='{"backend":"transformers"}'--save_dir=$SAVE_DIR
python run_eval.py "govtech/lionguard-2-lite" --type="hf" --kwargs='{"backend":"transformers"}'--save_dir=$SAVE_DIR
python run_eval.py "nvidia/llama-3.1-nemotron-safety-guard-8b-v3" --type="hf" --save_dir=$SAVE_DIR

