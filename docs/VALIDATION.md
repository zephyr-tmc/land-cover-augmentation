# 検証 / 验证 / Validation

Date: 2026-10-20. Platform: macOS arm64. Python: 3.12.14.

## Performed checks

- Parsed all 25 curated Python source files.
- Confirmed `--help` succeeds for the later classifier's two runners, attribute tagger, LoRA training wrapper, constrained generator and shared evaluator.
- Constructed an artificial four-class dataset with three 32×32 images per class in each of train, validation and test.
- Ran one training step of the later `Real-only` / `smallcnnpp` classifier on CPU, with one epoch and no background loader workers.
- Verified three validation-selected checkpoints and six nonempty metric/prediction export folders: validation and test for each selected checkpoint.
- Generated LoRA captions and an attribute report for three synthetic images using two synthetic reference sets.
- Inspected the report-derived result figure and checked its values against the thesis table.

The matching shared evaluation module was restored from the later source tree so that the later runner can import its metric functions. No original dataset or pretrained weights were needed for these checks.

## Scope

Synthetic images verify execution and file interfaces only. Their scores are not research results. The historical result table was not recomputed.

Full GPU LoRA training, Stable Diffusion generation, real-dataset classification, all earlier experimental variants and binary reproducibility were not tested. Source compatibility with future external sd-scripts revisions is not guaranteed.

The preserved research code emits CUDA AMP deprecation and CPU pin-memory warnings under the validation environment. They did not prevent the checked CPU training and exports.

Core package versions used in these checks are recorded in [requirements-validated.txt](../requirements-validated.txt). This records the smoke-check environment, rather than claiming to be the original thesis environment.
