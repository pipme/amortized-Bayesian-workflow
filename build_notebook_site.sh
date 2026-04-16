#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
NOTEBOOK_SRC="$REPO_ROOT/examples/workflow_step_by_step.ipynb"
OUT_DIR="$SCRIPT_DIR/notebooks"
NOTEBOOK_OUT_IPYNB="$OUT_DIR/workflow_step_by_step.ipynb"
JUPYTER_NO_CONFIG=1

mkdir -p "$OUT_DIR"

if [[ "${EXECUTE_NOTEBOOK:-0}" == "1" ]]; then
  # Execute into website output folder to avoid modifying source notebook.
  JUPYTER_NO_CONFIG="$JUPYTER_NO_CONFIG" jupyter nbconvert \
    --to notebook \
    --execute \
    --ExecutePreprocessor.timeout=1800 \
    "$NOTEBOOK_SRC" \
    --output workflow_step_by_step.ipynb \
    --output-dir "$OUT_DIR"

  HTML_INPUT="$NOTEBOOK_OUT_IPYNB"
else
  # Keep a same-origin ipynb copy so browser download works reliably.
  cp "$NOTEBOOK_SRC" "$NOTEBOOK_OUT_IPYNB"
  HTML_INPUT="$NOTEBOOK_SRC"
fi

JUPYTER_NO_CONFIG="$JUPYTER_NO_CONFIG" jupyter nbconvert \
  --to html \
  --template lab \
  "$HTML_INPUT" \
  --output workflow_step_by_step.html \
  --output-dir "$OUT_DIR"

echo "Rendered notebook to $OUT_DIR/workflow_step_by_step.html"
