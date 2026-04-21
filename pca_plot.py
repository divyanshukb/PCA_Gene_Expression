import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv('data/filtered.tsv.gz', sep='\t', compression='gzip')
df.columns = [str(c).strip() for c in df.columns]

labels = pd.read_csv('data/class.tsv', sep='\t', header=None, names=['label'])
labels = labels['label'].values   # 1 = ER+,  0 = ER-

print(f"Matrix shape : {df.shape}")          # (105, 16174)
print(f"ER+ : {(labels==1).sum()},  ER- : {(labels==0).sum()}")

# ── Extract XBP1 (ID=4404) and GATA3 (ID=4359) ────────────────────────────────
xbp1  = df['4404'].values.astype(float)
gata3 = df['4359'].values.astype(float)

er_pos = labels == 1
er_neg = labels == 0
colors = np.where(labels == 1, 'red', 'black')

# ── PCA on the 2-D standardized data ──────────────────────────────────────────
X2 = np.column_stack([gata3, xbp1])
X2_scaled = StandardScaler().fit_transform(X2)

pca2 = PCA(n_components=2)
pca2.fit(X2_scaled)
pc1_scores = X2_scaled @ pca2.components_[0]

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel a: scatter
ax = axes[0]
ax.scatter(gata3[er_neg], xbp1[er_neg], c='black', marker='s', s=30, label='ER⁻')
ax.scatter(gata3[er_pos], xbp1[er_pos], c='red',   marker='s', s=30, label='ER⁺')
ax.set_xlabel('GATA3', style='italic', fontsize=13)
ax.set_ylabel('XBP1',  style='italic', fontsize=13)
ax.set_title('a', fontweight='bold', loc='left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(frameon=False)

# Panel c: PC1 projection
ax = axes[1]
xmin = pc1_scores.min() - 0.5
rows = [
    ('All', 0,  pc1_scores,           colors),
    ('ER⁻',-1,  pc1_scores[er_neg],  ['black']*er_neg.sum()),
    ('ER⁺',-2,  pc1_scores[er_pos],  ['red']*er_pos.sum()),
]
for lbl, y, scores, col in rows:
    ax.axhline(y=y, color='gray', lw=0.8)
    ax.scatter(scores, np.full_like(scores, float(y)),
               c=col, marker='o', s=18, alpha=0.85)
    ax.text(xmin - 0.2, y, lbl, ha='right', va='center', fontsize=11)

ax.set_yticks([])
ax.set_xlabel('Projection onto PC1', fontsize=12)
ax.set_title('c', fontweight='bold', loc='left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_xlim(xmin - 1, pc1_scores.max() + 0.5)

plt.tight_layout()
plt.savefig('pca_result.png', dpi=150, bbox_inches='tight')
print("Saved → pca_result.png")

