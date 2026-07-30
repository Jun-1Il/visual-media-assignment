# 現行実験の成果物

このディレクトリは、Qwen3-VL / MLXでViCropの内部重要度手法を比較した
**現行の91画像・11 variant実験だけ**を収録する正本です。初期実験、
rel-att単独の途中実験、単一画像のdebug出力は
`archive/legacy_results_2026-07-30/` へ非破壊で退避しました。

## 実験単位

- モデル: `mlx-community/Qwen3-VL-2B-Instruct-4bit`
- 全画像: 91
- layer calibration: 8画像
- held-out評価: 83画像
- variant: 11
- 推論レコード: 91 × 11 = 1001行
- 選択層: 17
- ensemble層: 17, 21, 20

11 variantは次のとおりです。

```text
baseline
rel_att
grad_att
pure_grad
pure_grad_repo
rel_att_high
grad_att_high
pure_grad_high
pure_grad_repo_high
rel_att_ensemble
rel_att_plus
```

## 収録内容

| パス | 内容 | 完全性の基準 |
|---|---|---|
| `internal_methods_predictions.csv` | 全推論の正本 | 1001行、各variant 91行 |
| `internal_methods_summary.json` | 全91画像とheld-out 83画像の集計 | 11 variant |
| `internal_methods_metrics_by_condition.csv` | 実行時に生成したdomain・条件別集計 | 66行 |
| `internal_methods_analysis.json` | held-out主分析とfailure case | 11 variant |
| `internal_methods_analysis_by_condition.csv` | held-out条件別集計 | 66行 |
| `layer_calibration.csv` | 8画像 × 28層の較正結果 | 224行 |
| `layer_selection.json` | 層選択プロトコルと順位 | 選択層17 |
| `calibration_crops/` | 層較正時のcrop | 8画像 × 28層 = 224枚 |
| `attention_reconstruction_validation.json` | 明示attentionとnative出力の数値検証 | `passed: true` |
| `gradient_internal_validation.json` | grad-att / pure-gradの数値検証 | `passed: true` |
| `rel_att/` | baseline以外の内部map・crop・metadata | 10系統 × 91画像 |
| `figures/internal_methods_bw/` | 現行分析の白黒図 | 6枚 |

`rel_att/` は開発初期から引き継いだディレクトリ名ですが、内容は
rel-attだけではありません。現行実験の `grad_att`、`pure_grad`、
高解像度版、公開コード差分版、改善版を含む10系統すべてのraw成果物です。
この名前は再生成スクリプトとの互換性のため維持しています。

## 正本と派生物

数値の正本は `internal_methods_predictions.csv` です。
`internal_methods_summary.json` と `internal_methods_metrics_by_condition.csv`
は実験runnerが生成し、`internal_methods_analysis.json`、
`internal_methods_analysis_by_condition.csv`、白黒図6枚は分析scriptが
この正本から生成します。

次のコマンドで、ディレクトリ構成、CSVの行数、variant、sample集合、
JSON集計との一致、raw mapの91画像分、較正crop、白黒図、検証JSONを
まとめて検査できます。

```bash
.venv/bin/python scripts/verify_current_results.py
```

GitHub公開用bundleは容量を抑えるため、1001行の推論正本・集計・検証・図を
すべて含める一方、raw mapは本文図に必要な代表7組だけを含み、224枚の
較正cropは含めません。bundle内では次の縮約版検査を使います。

```bash
.venv/bin/python scripts/verify_current_results.py --bundle
```

## 再生成順

```bash
.venv/bin/python scripts/calibrate_rel_att_layers.py
.venv/bin/python scripts/run_internal_methods_experiment.py
.venv/bin/python scripts/analyze_internal_methods_results.py
.venv/bin/python scripts/validate_rel_att_internals.py
.venv/bin/python scripts/validate_gradient_internals.py
```

旧成果物は比較や監査のためarchiveに保存していますが、現行レポートの数値、
図、GitHub配布bundleの入力には使いません。
