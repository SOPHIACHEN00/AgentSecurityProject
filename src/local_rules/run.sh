#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
run=${1:?Pass a project-relative runs directory}
q3=${QWEN3_PYTHON:-python}
q35=${QWEN35_PYTHON:-.venv-q35/bin/python}
mkdir -p "$run"
exec 9>"$run/inference.lock"
flock -n 9
trap 'echo "$?" > "$run/inference-exit-code"' EXIT
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
"$q3" -Bu src/local_rules/prepare.py "$run"
for name in Qwen3-4B-Instruct-2507 Qwen3.5-0.8B Qwen3.5-2B Qwen3-8B; do
    if [[ "$name" == Qwen3.5-* ]]; then python_model="$q35"; else python_model="$q3"; fi
    "$python_model" -Bu src/local_rules/infer.py "$run" "$name"
done
"$q3" -Bu src/local_rules/report.py "$run"
