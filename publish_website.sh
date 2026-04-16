#!/usr/bin/env bash
set -euo pipefail

TARGET_BRANCH="gh-pages"
REMOTE="origin"
SKIP_BUILD=0
NO_COMMIT=0
SKIP_BRANCH_PUSH=0
EXECUTE_NOTEBOOK=1

usage() {
  cat <<'EOF'
Usage: website/publish_website.sh [options]

Builds website assets, optionally commits website/ changes,
pushes the current branch, and publishes website/ to gh-pages via git subtree.

Options:
  --skip-build        Do not run website/build_notebook_site.sh
  --no-commit         Do not create a commit before publishing
  --skip-branch-push  Do not push the current branch before publishing
  --execute-notebook  Execute notebook during build (default)
  --no-execute-notebook
                     Build site without executing notebook cells
  --target-branch BR  Publish to this branch (default: gh-pages)
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
    --skip-branch-push)
      SKIP_BRANCH_PUSH=1
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
    --target-branch)
      TARGET_BRANCH="${2:-}"
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
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

CURRENT_BRANCH="$(git branch --show-current)"
if [[ -z "$CURRENT_BRANCH" ]]; then
  echo "Not on a branch. Please switch to a branch and retry." >&2
  exit 1
fi

if [[ "$SKIP_BUILD" -eq 0 ]]; then
  EXECUTE_NOTEBOOK="$EXECUTE_NOTEBOOK" bash website/build_notebook_site.sh
fi

# Stage only website files; keep unrelated changes untouched.
git add website

# Some repository-wide ignore rules hide website assets (e.g., *.png, dotfiles).
# Force-add required website artifacts when present.
git add -f website/.nojekyll 2>/dev/null || true
git add -f website/static/images/*.png website/static/images/*.ico 2>/dev/null || true

if [[ "$NO_COMMIT" -eq 0 ]]; then
  if ! git diff --cached --quiet -- website; then
    git commit -m "Update website artifacts"
  fi
fi

if [[ "$SKIP_BRANCH_PUSH" -eq 0 ]]; then
  git push "$REMOTE" "$CURRENT_BRANCH"
fi

git subtree push --prefix website "$REMOTE" "$TARGET_BRANCH"

echo "Published website/ to ${REMOTE}/${TARGET_BRANCH} from ${CURRENT_BRANCH}"