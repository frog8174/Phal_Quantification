"""
Full-dataset color vectorization (§4.4.3).
Replicates ColorQuantify.ipynb Cell 5's per-flower logic EXACTLY, but over all
189 petal cutouts (the notebook demo used only 10). Uses the already-built K=6
palette (Pigment/global_centers_lab.npy). Writes Pigment/04_quantified_features_full.csv.
"""
import cv2, numpy as np, pandas as pd, glob, os
from scipy.spatial.distance import cdist

image_dir = '../Segmentation/Inference/petal_only_outputs/petal_cutouts/'
output_dir = './Pigment'
centers = np.load(os.path.join(output_dir, 'global_centers_lab.npy'))  # K=6 LAB centers (cv2 8-bit space)

paths = sorted(glob.glob(os.path.join(image_dir, '*.png')))
rows = []
for p in paths:
    name = os.path.basename(p).replace('.png', '')
    bgr = cv2.imread(p)
    if bgr is None:
        continue
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    a = lab[:, :, 1].astype(np.float32) - 128
    b = lab[:, :, 2].astype(np.float32) - 128
    chroma = np.sqrt(a ** 2 + b ** 2)
    l = lab[:, :, 0]
    valid = (l > 15) & (chroma > 10)          # same dual threshold as the notebook / §3.6
    vpx = lab[valid]
    if len(vpx) == 0:
        continue
    d = cdist(vpx, centers, metric='euclidean')
    ratios = np.bincount(np.argmin(d, axis=1), minlength=len(centers)) / len(vpx)
    row = {'Sample_ID': name}
    for i in range(len(centers)):
        row[f'Bin_{i}'] = ratios[i]
    rows.append(row)

df = pd.DataFrame(rows)
out = os.path.join(output_dir, '04_quantified_features_full.csv')
df.to_csv(out, index=False)

print(f'[OK] {len(df)} flowers -> {out}')
means = df.drop(columns=['Sample_ID']).mean()
print('mean bin proportions:', {k: round(v, 4) for k, v in means.items()})
# how many flowers are "dominated" by a single bin (>0.7)
bins = df.drop(columns=['Sample_ID'])
dom = (bins.max(axis=1) > 0.7).sum()
print(f'flowers with a single bin >70%: {dom}/{len(df)}')
print('per-bin: # flowers where that bin is the largest:',
      {c: int((bins.idxmax(axis=1) == c).sum()) for c in bins.columns})
