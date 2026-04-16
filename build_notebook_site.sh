#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="$SCRIPT_DIR/notebooks"
NOTEBOOK_OUT_IPYNB="$OUT_DIR/workflow_step_by_step.ipynb"
TMP_NOTEBOOK_SRC="$OUT_DIR/.workflow_step_by_step.main.ipynb"
MAIN_REMOTE="${MAIN_REMOTE:-origin}"
MAIN_BRANCH="${MAIN_BRANCH:-main}"
MAIN_NOTEBOOK_PATH="${MAIN_NOTEBOOK_PATH:-examples/workflow_step_by_step.ipynb}"
JUPYTER_NO_CONFIG=1

mkdir -p "$OUT_DIR"

# Pull the canonical notebook source directly from main.
git fetch "$MAIN_REMOTE" "$MAIN_BRANCH"
git show "$MAIN_REMOTE/$MAIN_BRANCH:$MAIN_NOTEBOOK_PATH" > "$TMP_NOTEBOOK_SRC"

if [[ "${EXECUTE_NOTEBOOK:-0}" == "1" ]]; then
  # Execute into the output folder to avoid mutating main's notebook.
  JUPYTER_NO_CONFIG="$JUPYTER_NO_CONFIG" jupyter nbconvert \
    --to notebook \
    --execute \
    --ExecutePreprocessor.timeout=1800 \
    "$TMP_NOTEBOOK_SRC" \
    --output workflow_step_by_step.ipynb \
    --output-dir "$OUT_DIR"

  HTML_INPUT="$NOTEBOOK_OUT_IPYNB"
else
  cp "$TMP_NOTEBOOK_SRC" "$NOTEBOOK_OUT_IPYNB"
  HTML_INPUT="$NOTEBOOK_OUT_IPYNB"
fi

JUPYTER_NO_CONFIG="$JUPYTER_NO_CONFIG" jupyter nbconvert \
  --to html \
  --template lab \
  "$HTML_INPUT" \
  --output workflow_step_by_step.html \
  --output-dir "$OUT_DIR"

rm -f "$TMP_NOTEBOOK_SRC"

echo "Rendered notebook to $OUT_DIR/workflow_step_by_step.html"
