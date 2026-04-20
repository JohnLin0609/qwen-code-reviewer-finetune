#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Part A: Run PurpleLlama CybersecEval instruct + autocomplete benchmarks
#
# Prerequisites:
#   1. Start the model server first (in a separate terminal):
#        source .venv/bin/activate
#        python eval/model_server.py --port 8000
#
#   2. Then run this script (uses the cyberseceval venv):
#        bash eval/run_cyberseceval.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$PROJECT_DIR/.venvs/cyberseceval"
PURPLE="$PROJECT_DIR/PurpleLlama"
DATASETS="$PURPLE/CybersecurityBenchmarks/datasets"
RESULTS_DIR="$PROJECT_DIR/eval/cyberseceval_results"

MODEL_NAME="qwen2.5-coder-7b-finetuned"
SERVER_URL="http://localhost:8000/v1/"
LLM_SPEC="OPENAI::${MODEL_NAME}::EMPTY::${SERVER_URL}"

mkdir -p "$RESULTS_DIR"

# Activate cyberseceval venv
source "$VENV/bin/activate"

# Check server is running
echo "Checking model server at $SERVER_URL ..."
if ! curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    echo "ERROR: Model server not running. Start it first:"
    echo "  source .venv/bin/activate"
    echo "  python eval/model_server.py --port 8000"
    exit 1
fi
echo "Server is running."

cd "$PURPLE"

# ── Instruct benchmark ──────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Running INSTRUCT benchmark (1681 prompts, all languages)"
echo "═══════════════════════════════════════════════════════════════"
python -m CybersecurityBenchmarks.benchmark.run \
    --benchmark=instruct \
    --prompt-path="$DATASETS/instruct/instruct-v2.json" \
    --response-path="$RESULTS_DIR/instruct_responses.json" \
    --stat-path="$RESULTS_DIR/instruct_stat.json" \
    --llm-under-test="$LLM_SPEC" \
    --run-llm-in-parallel=1 \
    --num-test-cases=0

echo ""
echo "Instruct results saved to: $RESULTS_DIR/instruct_stat.json"

# ── Autocomplete benchmark ──────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Running AUTOCOMPLETE benchmark (1916 prompts)"
echo "═══════════════════════════════════════════════════════════════"
python -m CybersecurityBenchmarks.benchmark.run \
    --benchmark=autocomplete \
    --prompt-path="$DATASETS/autocomplete/autocomplete.json" \
    --response-path="$RESULTS_DIR/autocomplete_responses.json" \
    --stat-path="$RESULTS_DIR/autocomplete_stat.json" \
    --llm-under-test="$LLM_SPEC" \
    --run-llm-in-parallel=1 \
    --num-test-cases=0

echo ""
echo "Autocomplete results saved to: $RESULTS_DIR/autocomplete_stat.json"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  All benchmarks complete!"
echo "  Results directory: $RESULTS_DIR"
echo "═══════════════════════════════════════════════════════════════"
