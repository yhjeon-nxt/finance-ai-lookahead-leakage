#!/usr/bin/env bash
# Build the submittable PDF from report/research_report.md.
#
#   deps : pandoc + weasyprint   (macOS: brew install pandoc weasyprint)
#   usage: bash report/build_pdf.sh   ->   report/research_report.pdf
#
# Notes:
#  * The stylesheet (report/pdf.css) is inlined into the HTML <head> rather than
#    linked — a linked stylesheet triggered a WeasyPrint per-glyph fallback quirk
#    that dropped U+2194 (↔). Inlining renders all symbols correctly.
#  * weasyprint runs with base-url = report/ so figures/*.png resolve.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

{ echo '<style>'; cat report/pdf.css; echo '</style>'; } > "$TMP/header.html"

pandoc report/research_report.md \
  --from gfm --standalone --toc --toc-depth=2 \
  --metadata title="Look-ahead Leakage in Open-Source LLM Trading Agents" \
  --include-in-header="$TMP/header.html" \
  -o "$TMP/report.html"

weasyprint -u "$ROOT/report/" "$TMP/report.html" report/research_report.pdf
echo "wrote report/research_report.pdf"
