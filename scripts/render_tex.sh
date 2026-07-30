#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if ! command -v tectonic >/dev/null 2>&1; then
  printf '%s\n' \
    "エラー: tectonic が見つかりません。" \
    "macOS + Homebrew: brew install tectonic" \
    "インストール後、もう一度 bash scripts/render_tex.sh を実行してください。" >&2
  exit 1
fi

if [[ ! -f report/student_info.tex ]]; then
  printf '%s\n' "エラー: report/student_info.tex が見つかりません。" >&2
  exit 1
fi

mkdir -p output/pdf
tectonic report/vicrop_mlx_report_tex.tex \
  --outdir output/pdf \
  --keep-logs

output_pdf="output/pdf/vicrop_mlx_report_tex.pdf"
if [[ ! -s "$output_pdf" ]]; then
  printf '%s\n' "エラー: PDFを生成できませんでした。" >&2
  exit 1
fi

printf '生成完了: %s/%s\n' "$repo_root" "$output_pdf"
