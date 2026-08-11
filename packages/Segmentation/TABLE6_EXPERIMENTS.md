# Table 6 — segmentation comparison (prepared 2026-06-28)

Goal: prove **CoordConv-X** (main model) on a clean controlled comparison.
All models: **train on `v2_dataset_split` (20/4), test on `eval-dataset` (9 held-out, leakage-checked), HFlip removed.**

## The 4 models
| Model | Train script → save_dir | Eval script → output | GPU |
|---|---|---|---|
| **CoordConv-X (proposed)** | `fine-tuning_coordconv.py` → `CoordConvX_v1` ✅ trained | `evaluation_coordconv.py` → `CoordConvX_eval-dataset` ✅ (re-run for CM csv) | cuda:3 / cuda:2 |
| **Plain DINOv3 (ablation)** | `fine-tuning.py` → `Finalexp_v1` (you're training) | `evaluation.py` → `Finalexp_v1_eval-dataset` | cuda:3 / cuda:2 |
| DeepLabV3 ResNet-50 | `baseline_deeplabv3.py` → `baseline_deeplabv3_v2` | `evaluation_deeplabv3.py` → `baseline_deeplabv3_eval-dataset` | cuda:2 |
| U-Net (ResNet-34, ImageNet) | `baseline_unet.py` → `Finalexp_baseline_unet_resnet34_v1` | `evaluation_unet.py` → `baseline_unet_eval-dataset` | cuda:2 |

> Plain vs CoordConv is the **ablation** (only difference = x-coord channel; recipe matched: epochs 100, patience 25, lr_head 2e-4, lr_backbone 5e-6, no HFlip).
> DeepLabV3 / U-Net are **architecture baselines** (same split + same test, own optimizers).

## What I changed (2026-06-28)
- `baseline_deeplabv3.py`: dataset → `v2_dataset_split`, save_dir → `baseline_deeplabv3_v2`, **removed HFlip** (was corrupting L/R petals → unfairly low baseline).
- `baseline_unet.py`: **rebuilt as a U-Net with an ImageNet-pretrained ResNet-34 encoder** (2026-06-29; the from-scratch version failed at 0.258). Differential LR (encoder 1e-5 / decoder 1e-4), HFlip removed, decodes to 1/2-res then bilinear-upsamples, gradient-checkpointed for 2400px. `evaluation_unet.py` model updated to match.
- `evaluation.py`: → plain `Finalexp_v1` checkpoint, `extract_layers=[-1]`, test = eval-dataset, + raw confusion-matrix CSV.
- `evaluation_deeplabv3.py`: → new checkpoint + eval-dataset, + CM CSV.
- `evaluation_coordconv.py`: + CM CSV (paths already correct).
- **new** `evaluation_unet.py`, **new** `build_table6.py`.
- `fine-tuning.py`: you already set save_dir=`Finalexp_v1`, patience=25. ✅

## Run order
**STATUS 2026-06-29:** DINOv3 (plain, main model) ✅, DeepLabV3 ✅ trained + evaluated. CoordConv-X **dropped** (did not beat plain). **Only the pretrained U-Net remains.**

Only-remaining step — train + evaluate the new pretrained U-Net, then rebuild the table:
```bash
# venv: SAM env (or `uv run` — both have torchvision; ResNet-34 weights download on first run, needs internet)
PY=/home/nas2/Workspace/Aaron/server2_envs/SAM/.venv/bin/python
$PY baseline_unet.py         # trains → training_result/Finalexp_baseline_unet_resnet34_v1  (device cuda:2 in CONFIG)
$PY evaluation_unet.py       # evaluates → Evaluation/baseline_unet_eval-dataset  (device cuda:3 in CONFIG)
$PY build_table6.py          # reprints Table 6 with the new U-Net row
```
The DINOv3/DeepLabV3 rows are already final in `Evaluation/{Finalexp_v1,baseline_deeplabv3}_eval-dataset/`; only the U-Net column will change.
This prints the markdown Table 6 (Global Pixel Acc / mAcc / mIoU / per-class IoU) **and** the
Petal_L↔Petal_R confusion row — the direct evidence for CoordConv-X.

## ⚠️ Watch-outs
- **The L/R confusion row is the headline**, not mIoU (all models score ~0.99 on petals). If CoordConv cuts L↔R swaps vs plain, that's the result.
- Removing HFlip will likely **raise** DeepLabV3/U-Net petal IoU vs the old paper numbers (old DeepLab petals = 0.835 were inflated-low by the HFlip bug). That's expected and more honest — the VFM advantage should still hold, just with a smaller, defensible petal gap.
- Check each eval's stdout `GT labels` vs `Pred labels` to confirm masks load correctly.
- `eval-dataset` = 9 images, so per-class means are sensitive to single outliers (e.g. 744A2809 labellum). Report per-image CSV ranges too.
