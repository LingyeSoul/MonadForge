#!/usr/bin/env bash
# Build the anima_lora image from a git ref (tracked files only — no models,
# datasets, outputs or uncommitted edits reach the image).
#
#   docker/build.sh [ref] [tag]      defaults: ref=main  tag=anima-lora:<ref short sha>
set -euo pipefail

REF="${1:-main}"
REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
SHA="$(git -C "$REPO_ROOT" rev-parse --short "$REF")"
TAG="${2:-anima-lora:$SHA}"

CTX="$(mktemp -d)"
trap 'rm -rf "$CTX"' EXIT

git -C "$REPO_ROOT" archive "$REF" | tar -x -C "$CTX"
# The Dockerfile/entrypoint come from the working tree, so they need not be
# committed on <ref> yet.
mkdir -p "$CTX/docker"
cp "$REPO_ROOT/docker/entrypoint.sh" "$CTX/docker/"

docker build -f "$REPO_ROOT/docker/Dockerfile" -t "$TAG" "$CTX"
echo "built $TAG from $REF ($SHA)"
