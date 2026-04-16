#!/usr/bin/env bash
set -euo pipefail

TARGET_BRANCH="gh-pages"
REMOTE="origin"
SOURCE_BRANCH="website-src"
BASE_BRANCH="main"
SKIP_BUILD=0
NO_COMMIT=0
SKIP_SOURCE_PUSH=0
SKIP_REBASE=0
ALLOW_NON_WEBSITE_DIFF=0
EXECUTE_NOTEBOOK=1
RESET_SOURCE_TO_BASE=0

usage() {
  cat <<'EOF'
Usage: website/publish_website.sh [options]

Builds website assets, optionally commits website/ changes,
pushes website-src, and publishes website/ to gh-pages via git subtree.

Options:
  --skip-build        Do not run website/build_notebook_site.sh
  --no-commit         Do not create a commit before publishing
  --skip-source-push  Do not push website-src before publishing
  --skip-rebase       Do not fetch/rebase website-src onto origin/<base-branch>
  --execute-notebook  Execute notebook during build (default)
  --no-execute-notebook
                     Build site without executing notebook cells
  --reset-source-to-main
                     Reset source branch to origin/<base-branch> and keep only website/
  --allow-non-website-diff
                     Allow differences from origin/<base-branch> outside website/
  --target-branch BR  Publish to this branch (default: gh-pages)
  --base-branch BR    Rebase source branch onto origin/<base-branch> (default: main)
  --source-branch BR  Required current branch (default: website-src)
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
    --skip-source-push)
      SKIP_SOURCE_PUSH=1
      shift
      ;;
    --skip-rebase)
      SKIP_REBASE=1
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
    --reset-source-to-main)
      RESET_SOURCE_TO_BASE=1
      shift
      ;;
    --allow-non-website-diff)
      ALLOW_NON_WEBSITE_DIFF=1
      shift
      ;;
    --target-branch)
      TARGET_BRANCH="${2:-}"
      shift 2
      ;;
    --base-branch)
      BASE_BRANCH="${2:-}"
      shift 2
      ;;
    --source-branch)
      SOURCE_BRANCH="${2:-}"
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
# Ensure source branch exists and is checked out.
if ! git show-ref --verify --quiet "refs/heads/$SOURCE_BRANCH"; then
  git fetch "$REMOTE"
  git switch "$BASE_BRANCH"
  git switch -c "$SOURCE_BRANCH"
fi

if [[ "$CURRENT_BRANCH" != "$SOURCE_BRANCH" ]]; then
  git switch "$SOURCE_BRANCH"
fi

if [[ "$RESET_SOURCE_TO_BASE" -eq 1 ]]; then
  if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Working tree is not clean. Commit/stash changes before --reset-source-to-main." >&2
    exit 1
  fi

  git fetch "$REMOTE"
  PRE_RESET_HEAD="$(git rev-parse HEAD)"
  git reset --hard "$REMOTE/$BASE_BRANCH"

  if git ls-tree --name-only "$PRE_RESET_HEAD" website >/dev/null 2>&1; then
    if git ls-tree --name-only "$PRE_RESET_HEAD" website | grep -q '^website$'; then
      git checkout "$PRE_RESET_HEAD" -- website
    fi
  fi

  git add website
  git add -f website/.nojekyll 2>/dev/null || true
  git add -f website/static/images/*.png website/static/images/*.ico 2>/dev/null || true

  if ! git diff --cached --quiet -- website; then
    git commit -m "Reset ${SOURCE_BRANCH} to ${REMOTE}/${BASE_BRANCH} and keep website/"
  fi

  SKIP_REBASE=1
fi

if [[ "$SKIP_REBASE" -eq 0 ]]; then
  git fetch "$REMOTE"
  git rebase "$REMOTE/$BASE_BRANCH"
fi

if [[ "$ALLOW_NON_WEBSITE_DIFF" -eq 0 ]]; then
  NON_WEBSITE_DIFF="$(git diff --name-only "$REMOTE/$BASE_BRANCH"..HEAD -- . ':(exclude)website/**')"
  if [[ -n "$NON_WEBSITE_DIFF" ]]; then
    echo "website-src differs from $REMOTE/$BASE_BRANCH outside website/:" >&2
    echo "$NON_WEBSITE_DIFF" >&2
    echo "Abort. Rebase/sync branch or use --allow-non-website-diff if intentional." >&2
    exit 1
  fi
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

if [[ "$SKIP_SOURCE_PUSH" -eq 0 ]]; then
  git push "$REMOTE" "$SOURCE_BRANCH"
fi

git subtree push --prefix website "$REMOTE" "$TARGET_BRANCH"

echo "Published website/ to ${REMOTE}/${TARGET_BRANCH} from ${SOURCE_BRANCH}"