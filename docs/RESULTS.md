# 実験記録 / 实验记录 / Reported experiments

## Provenance

The main README table is transcribed from the author's graduation thesis, Chapter “実験”, table “分類性能の比較（竹林クラス追加数=224枚）”, source label `tab:main_results`. The source document was available locally during preparation. It is not redistributed as part of this code excerpt.

The earlier progress reports dated 2025-10-10 and 2025-10-23 describe a different experimental stage. Their validation accuracies and Macro-F1 values have not been merged into the thesis comparison.

## Thesis dataset description

The thesis describes four aerial-image classes with an 8:1:1 train/validation/test split:

| Class | Train | Validation | Test | Total |
| --- | ---: | ---: | ---: | ---: |
| bamboo | 376 | 47 | 47 | 470 |
| pine | 816 | 102 | 102 | 1020 |
| rice | 1768 | 221 | 221 | 2210 |
| tea | 136 | 17 | 17 | 170 |
| Total | 3096 | 387 | 387 | 3870 |

The source is the [GSI dataset](https://gisstar.gsi.go.jp/gsi-dataset/index_ja.html). Neither the images nor the original split manifests are included here. The public code excerpt alone cannot verify geographic independence or the absence of related patches across splits.

## Main comparison as reported

| Method | Macro-F1 | Macro-AP | ROC-AUC |
| --- | --- | --- | --- |
| Real-only | 0.9244 ± 0.0048 | 0.9772 ± 0.0024 | 0.9961 ± 0.0009 |
| Oversampling | 0.9347 ± 0.0131 | 0.9807 ± 0.0058 | 0.9967 ± 0.0010 |
| Traditional DA | 0.9343 ± 0.0127 | 0.9803 ± 0.0045 | 0.9966 ± 0.0011 |
| LSGAN | 0.9397 ± 0.0077 | 0.9843 ± 0.0046 | 0.9976 ± 0.0006 |
| LoRA | 0.9363 ± 0.0048 | 0.9829 ± 0.0045 | 0.9971 ± 0.0006 |
| LoRA-align | 0.9486 ± 0.0042 | 0.9870 ± 0.0022 | 0.9977 ± 0.0004 |

The ± values are copied as reported. The inspected source does not specify their run count or whether they represent standard deviation, standard error or another quantity, so no additional statistical interpretation is made. Machine-readable values are in [reported-results.csv](reported-results.csv).

All augmented conditions add 224 bamboo images, from 376 to 600 bamboo training samples. This does not make all four classes equal in size. The reported Macro-F1 difference between Real-only and LoRA-align is 0.0242, or 2.42 percentage points on a 0–100 scale.

The figure displays reported central values only. It is a visualization of the thesis table, not a new evaluation.

## Zoom ablation as reported

The thesis table `tab:ablation_zoom` reports Macro-F1 of `0.9391 ± 0.0121` for LoRA-zoom and `0.9486 ± 0.0042` for LoRA-align. This is a historical ablation result; it was not rerun during preparation.

## Code provenance and changes for publication

- `CNN_model/`: selected earlier scripts from the author's local land-cover project.
- `classifier/`: later baseline implementation, plus the later classification runner from the final research pipeline.
- `lora/`: captioning, training-wrapper, constrained-generation and feature-visualization scripts from the final pipeline.
- `evaluate.py`: shared evaluation utilities placed at the parent path expected by the later classification runner.
- Original image archives, trained models, location lists, copied third-party source trees and unrelated student work were excluded.
- The local img2img client's two workstation paths were replaced with environment-configurable relative defaults.
- A source-to-publication file map is in [source-manifest.json](source-manifest.json).

The LoRA training wrapper delegates training to sd-scripts. Foundation models, public data and external implementations are not claimed as original work.

## Reproducibility limits

The public excerpt has not rerun the GPU training or the full original evaluation. The original environment lockfile, complete per-run logs, split manifests and model checkpoints are unavailable in this publication. The thesis settings and public script defaults also differ in places; the source snapshot should be treated as research code with documented execution examples, rather than an exact numerical reproduction package.
