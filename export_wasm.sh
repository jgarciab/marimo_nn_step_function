#!/usr/bin/env bash
# Build a static WASM bundle for local preview.
#
# This helper writes to build/, which is generated from app.py.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export UV_LINK_MODE=copy

OUT_DIR="${1:-build}"
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

echo "Exporting WASM bundle to $OUT_DIR/index.html ..."
uv run python -m marimo export html-wasm app.py -o "$OUT_DIR" --mode run -f
test -f "$OUT_DIR/index.html"

echo
echo "Done. To preview locally:"
echo "  cd $OUT_DIR && python -m http.server 8000"
echo "  then open http://localhost:8000/"
