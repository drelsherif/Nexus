#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# git_push_versioned.sh — Push NEXUS updates to GitHub with version tags
#
# Usage:
#   bash git_push_versioned.sh "Add Apex Learner"         # auto-version patch
#   bash git_push_versioned.sh "Add Apex Learner" minor   # bump minor version
#   bash git_push_versioned.sh "Add Apex Learner" major   # bump major version
#   bash git_push_versioned.sh "Add Apex Learner" v3.1.0  # explicit tag
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

MSG="${1:-Update NEXUS}"
BUMP="${2:-patch}"   # patch | minor | major | or explicit tag like v3.1.0

cd "$(dirname "$0")"

# ── 1. Show what will be committed ──────────────────────────────────────────
echo ""
echo "── Staged changes ──────────────────────────────────────"
git status --short
echo ""

# ── 2. Stage all changes ────────────────────────────────────────────────────
git add -A

# ── 3. Determine next version tag ──────────────────────────────────────────
if [[ "${BUMP}" == v*.*.* ]]; then
    # Explicit tag provided
    NEW_TAG="${BUMP}"
else
    # Find the latest semver tag
    LATEST=$(git tag -l 'v*.*.*' | sort -V | tail -1)
    if [[ -z "${LATEST}" ]]; then
        LATEST="v0.0.0"
    fi

    # Parse major.minor.patch
    MAJOR=$(echo "${LATEST}" | cut -d. -f1 | tr -d v)
    MINOR=$(echo "${LATEST}" | cut -d. -f2)
    PATCH=$(echo "${LATEST}" | cut -d. -f3)

    case "${BUMP}" in
        major) MAJOR=$((MAJOR+1)); MINOR=0; PATCH=0 ;;
        minor) MINOR=$((MINOR+1)); PATCH=0 ;;
        patch) PATCH=$((PATCH+1)) ;;
        *)     echo "[Error] Unknown bump type: ${BUMP}"; exit 1 ;;
    esac

    NEW_TAG="v${MAJOR}.${MINOR}.${PATCH}"
fi

echo "── Version: ${NEW_TAG} ─────────────────────────────────"

# ── 4. Commit ────────────────────────────────────────────────────────────────
git commit -m "${MSG} [${NEW_TAG}]"

# ── 5. Tag the commit ────────────────────────────────────────────────────────
git tag -a "${NEW_TAG}" -m "${MSG}"

# ── 6. Push commit + tag ─────────────────────────────────────────────────────
git push origin main
git push origin "${NEW_TAG}"

echo ""
echo "✓ Pushed commit + tag ${NEW_TAG} to https://github.com/drelsherif/Nexus"
echo ""

# ── 7. Show tag history ──────────────────────────────────────────────────────
echo "── Version history ─────────────────────────────────────"
git tag -l 'v*.*.*' | sort -V | tail -10 | while read t; do
    DATE=$(git log -1 --format="%ai" "${t}" | cut -d' ' -f1)
    MSG=$(git tag -l -n1 "${t}" | sed "s/^${t}[[:space:]]*//" | cut -c1-60)
    echo "  ${t}  ${DATE}  ${MSG}"
done
echo ""
