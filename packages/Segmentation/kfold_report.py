"""
K-Fold report aggregator
========================
讀取各折 evaluation 的 summary.json，算 mean ± std，輸出論文用的 CV 結果。

用法：
    uv run kfold_report.py --glob "./Evaluation/CoordConvX_FOLD*/summary.json"
    uv run kfold_report.py --glob "./Evaluation/CoordConvX_FOLD*/summary.json" --out cv_report.json
"""

import argparse
import glob
import json
import numpy as np

CLASS_NAMES = [
    "Background", "Column", "Dorsal Sepal", "Labellum",
    "Lateral Sepal", "Petal", "Petal_L", "Petal_R"
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="各折 summary.json 的 glob pattern")
    ap.add_argument("--out", default=None, help="輸出彙整 json (選填)")
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    if not files:
        print(f"找不到任何 summary.json：{args.glob}")
        return

    print(f"找到 {len(files)} 折：")
    for f in files:
        print(f"  - {f}")

    miou, macc, pacc = [], [], []
    per_class = {c: [] for c in CLASS_NAMES}
    for f in files:
        d = json.load(open(f))
        miou.append(d["mIoU"]); macc.append(d["mAcc"]); pacc.append(d["Global Pixel Acc"])
        for c in CLASS_NAMES:
            per_class[c].append(d["IoU_per_class"][c])

    def ms(xs):
        return float(np.mean(xs)), float(np.std(xs))

    print("\n=== K-Fold CV (mean ± std) ===")
    print(f"mIoU            : {np.mean(miou):.4f} ± {np.std(miou):.4f}")
    print(f"mAcc            : {np.mean(macc):.4f} ± {np.std(macc):.4f}")
    print(f"Global PixelAcc : {np.mean(pacc):.4f} ± {np.std(pacc):.4f}")
    print("\nPer-class IoU (mean ± std):")
    for c in CLASS_NAMES:
        m, s = ms(per_class[c])
        print(f"  {c:<14}: {m:.4f} ± {s:.4f}")

    # 特別點出左右花瓣 — 本次改動的重點
    print("\n>>> 重點：Petal_L / Petal_R")
    for c in ("Petal_L", "Petal_R"):
        m, s = ms(per_class[c])
        print(f"  {c}: {m:.4f} ± {s:.4f}")

    if args.out:
        report = {
            "n_folds": len(files),
            "mIoU": ms(miou), "mAcc": ms(macc), "Global Pixel Acc": ms(pacc),
            "IoU_per_class": {c: ms(per_class[c]) for c in CLASS_NAMES},
        }
        json.dump(report, open(args.out, "w"), indent=4)
        print(f"\n已輸出 -> {args.out}")


if __name__ == "__main__":
    main()
