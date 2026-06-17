#!/usr/bin/env bash
# Build the submittable PDFs from the report markdown.
#
#   deps : pandoc + weasyprint   (macOS: brew install pandoc weasyprint)
#   usage: bash report/build_pdf.sh
#     -> report/research_report.pdf        (full report)
#     -> report/research_report_short.pdf  (2-page short version)
#
# Notes:
#  * pdf.css is inlined into the HTML <head> rather than linked — a linked stylesheet
#    triggered a WeasyPrint per-glyph fallback quirk that dropped U+2194 (↔). Inlining
#    renders all symbols correctly.
#  * weasyprint runs with base-url = report/ so figures/*.png resolve.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

build() {   # <md> <css> <out> <title> [extra pandoc args...]
  local md="$1" css="$2" out="$3" title="$4"; shift 4
  { echo '<style>'; cat "$css"; echo '</style>'; } > "$TMP/header.html"
  pandoc "$md" --from gfm --standalone --metadata title="$title" \
    --include-in-header="$TMP/header.html" "$@" -o "$TMP/doc.html"
  weasyprint -u "$ROOT/report/" "$TMP/doc.html" "$out"
  echo "wrote $out"
}

build report/research_report.md       report/pdf.css       report/research_report.pdf \
      "Look-ahead Leakage in Open-Source LLM Trading Agents" --toc --toc-depth=2

build report/research_report_short.md report/pdf_short.css report/research_report_short.pdf \
      "Look-ahead Leakage in Open-Source LLM Trading Agents — Short Report"
