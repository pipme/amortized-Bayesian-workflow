#!/usr/bin/env bash
set -euo pipefail

TARGET_BRANCH="gh-pages"
REMOTE="origin"
MAIN_BRANCH="main"
SKIP_BUILD=0
NO_COMMIT=0
EXECUTE_NOTEBOOK=0

usage() {
  cat <<'EOF'
Usage: publish_website.sh [options]

Builds website assets on gh-pages, commits changes, and pushes gh-pages.

Options:
  --skip-build        Do not run website/build_notebook_site.sh
  --no-commit         Do not create a commit before publishing
  --execute-notebook  Execute notebook during build
  --no-execute-notebook
                     Build site without executing notebook cells (default)
  --main-branch BR    Pull notebook source from this branch (default: main)
  --remote NAME       Push to this remote (default: origin)
  -h, --help          Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-build)
      SKIP_BUILD=1
      shift
      ;;
    --no-commit)
      NO_COMMIT=1
      shift
      ;;
    --execute-notebook)
      EXECUTE_NOTEBOOK=1
      shift
      ;;
    --no-execute-notebook)
      EXECUTE_NOTEBOOK=0
      shift
      ;;
    --main-branch)
      MAIN_BRANCH="${2:-}"
      shift 2
      ;;
    --remote)
      REMOTE="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR" && pwd)"

cd "$REPO_ROOT"

CURRENT_BRANCH="$(git branch --show-current)"
if [[ -z "$CURRENT_BRANCH" ]]; then
  echo "Not on a branch. Please switch to a branch and retry." >&2
  exit 1
fi

if [[ "$CURRENT_BRANCH" != "$TARGET_BRANCH" ]]; then
  echo "This script is gh-pages-only. Switch to '$TARGET_BRANCH' and retry." >&2
  exit 1
fi

if [[ "$SKIP_BUILD" -eq 0 ]]; then
  EXECUTE_NOTEBOOK="$EXECUTE_NOTEBOOK" MAIN_REMOTE="$REMOTE" MAIN_BRANCH="$MAIN_BRANCH" bash "$SCRIPT_DIR/build_notebook_site.sh"
fi

git add -A

if [[ "$NO_COMMIT" -eq 0 ]]; then
  if ! git diff --cached --quiet; then
    git commit -m "Update website from ${REMOTE}/${MAIN_BRANCH}"
  fi
fi

git push "$REMOTE" "$TARGET_BRANCH"

echo "Published ${TARGET_BRANCH} from ${REMOTE}/${MAIN_BRANCH} notebook source"