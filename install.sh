#!/usr/bin/env sh
# anima_lora bootstrap installer (Linux / macOS).
#
#   curl -LsSf https://raw.githubusercontent.com/sorryhyun/anima_lora/main/install.sh | sh
#
# Installs uv if missing, downloads the latest release tarball (no git
# required), seeds the update baseline so the first `make update` is clean,
# and runs `uv sync`. The resolve-latest / tarball / manifest logic mirrors
# scripts/update.py — keep the two in sync.
#
# Options (env vars, since args are awkward through a pipe):
#   ANIMA_VERSION=v1.4.0   install a specific tag        (default: latest)
#   ANIMA_DIR=./somewhere  target directory              (default: ./anima_lora)
# Or with explicit args:  sh -s -- [version] [dir]
set -eu

REPO="sorryhyun/anima_lora"
VERSION="${ANIMA_VERSION:-${1:-}}"
DIR="${ANIMA_DIR:-${2:-anima_lora}}"

say()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

command -v curl >/dev/null 2>&1 || die "curl is required"
command -v tar  >/dev/null 2>&1 || die "tar is required"

# 1. uv ----------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  say "installing uv (https://astral.sh/uv)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # Make uv visible to this shell without a re-login.
  # shellcheck disable=SC1090
  [ -f "$HOME/.local/bin/env" ] && . "$HOME/.local/bin/env"
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || die "uv install failed; open a new shell and re-run"

# 2. resolve the release tag -------------------------------------------------
if [ -z "$VERSION" ]; then
  say "resolving latest release of $REPO"
  VERSION=$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" \
    | grep -m1 '"tag_name"' \
    | sed -E 's/.*"tag_name"[[:space:]]*:[[:space:]]*"([^"]+)".*/\1/')
  [ -n "$VERSION" ] || die "could not resolve latest release tag from GitHub API"
fi
say "installing $REPO @ $VERSION → $DIR/"

[ -e "$DIR" ] && [ -n "$(ls -A "$DIR" 2>/dev/null)" ] && \
  die "$DIR/ already exists and is not empty — pass a different ANIMA_DIR"

# 3. download + extract ------------------------------------------------------
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
TARBALL="https://github.com/$REPO/archive/refs/tags/$VERSION.tar.gz"
say "downloading $TARBALL"
curl -LsSf "$TARBALL" -o "$TMP/release.tar.gz" || die "download failed"
tar -xzf "$TMP/release.tar.gz" -C "$TMP"
# GitHub source tarballs contain a single top-level dir (anima_lora-<tag>).
TOP=$(find "$TMP" -mindepth 1 -maxdepth 1 -type d ! -name '.*' | head -n1)
[ -n "$TOP" ] || die "unexpected tarball layout"
mkdir -p "$DIR"
# move contents (including dotfiles) into the target dir
(cd "$TOP" && tar -cf - .) | (cd "$DIR" && tar -xf -)

cd "$DIR"

# 4. seed the update baseline (before uv sync, so .venv doesn't get hashed) ---
say "seeding update baseline (.anima_release.json)"
uv run --no-project python scripts/update.py --seed-manifest --version "$VERSION" \
  || say "manifest seed skipped (first \`make update\` will back up instead — harmless)"

# 5. dependencies ------------------------------------------------------------
say "running uv sync (this resolves torch + flash-attn; may take a while)"
uv sync

cat <<EOF

$(printf '\033[1;32m✓ installed to %s/\033[0m' "$DIR")

Next steps:
  cd $DIR
  hf auth login            # authenticate for gated model downloads
  make download-models     # DiT + Qwen3 text encoder + VAE into models/
  make gui                 # or:  make lora   (CLI training)

Update later with:  make update
EOF
