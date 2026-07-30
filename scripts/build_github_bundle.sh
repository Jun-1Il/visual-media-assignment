#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
target="$repo_root/github_upload/vicrop-mlx-assignment"
stage="$(mktemp -d "${TMPDIR:-/tmp}/vicrop-github.XXXXXX")"

cleanup() {
  rm -rf -- "$stage"
}
trap cleanup EXIT

mkdir -p \
  "$stage/src" \
  "$stage/tests" \
  "$stage/scripts" \
  "$stage/report" \
  "$stage/data" \
  "$stage/results/figures" \
  "$stage/output/pdf"

rsync -a \
  "$repo_root/README.md" \
  "$repo_root/EXPERIMENT_REPORT.md" \
  "$repo_root/AI_USAGE_LOG.md" \
  "$repo_root/pyproject.toml" \
  "$repo_root/.gitignore" \
  "$stage/"
rsync -a --exclude '__pycache__/' "$repo_root/src/" "$stage/src/"
rsync -a --exclude '__pycache__/' "$repo_root/tests/" "$stage/tests/"
rsync -a \
  --exclude '__pycache__/' \
  --exclude 'build_report.py' \
  --exclude 'run_experiment.py' \
  --exclude 'analyze_results.py' \
  "$repo_root/scripts/" \
  "$stage/scripts/"
rsync -a \
  "$repo_root/report/student_info.tex" \
  "$repo_root/report/vicrop_mlx_report_tex.tex" \
  "$repo_root/report/vicrop_mlx_report.md" \
  "$stage/report/"
rsync -a "$repo_root/data/manifest.csv" "$stage/data/"
rsync -a "$repo_root/synthetic/" "$stage/synthetic/"
rsync -a "$repo_root/imagegen/" "$stage/imagegen/"

for result_name in \
  README.md \
  internal_methods_predictions.csv \
  internal_methods_summary.json \
  internal_methods_metrics_by_condition.csv \
  internal_methods_analysis.json \
  internal_methods_analysis_by_condition.csv \
  layer_calibration.csv \
  layer_selection.json \
  attention_reconstruction_validation.json \
  gradient_internal_validation.json
do
  rsync -a "$repo_root/results/$result_name" "$stage/results/"
done
rsync -a \
  "$repo_root/results/figures/internal_methods_bw/" \
  "$stage/results/figures/internal_methods_bw/"

# Only the representative raw maps needed to rebuild the report figures are
# published.  All 91 x 10 maps can be regenerated from the prediction script
# and would make the GitHub repository unnecessarily large.
for example in \
  "rel_att:synthetic__002_small_clear" \
  "grad_att:synthetic__002_small_clear" \
  "pure_grad:synthetic__002_small_clear" \
  "pure_grad_repo:synthetic__002_small_clear" \
  "rel_att:synthetic__004_tiny_clear" \
  "rel_att:synthetic__026_multi_region_relation" \
  "rel_att_plus:synthetic__026_multi_region_relation"
do
  variant="${example%%:*}"
  sample="${example#*:}"
  mkdir -p "$stage/results/rel_att/$variant"
  rsync -a \
    "$repo_root/results/rel_att/$variant/$sample/" \
    "$stage/results/rel_att/$variant/$sample/"
done

rsync -a \
  "$repo_root/output/pdf/vicrop_mlx_report_tex.pdf" \
  "$stage/output/pdf/"

mkdir -p "$target"
rsync -a --delete "$stage/" "$target/"

printf 'GitHub bundle: %s\n' "$target"
