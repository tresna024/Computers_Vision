import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ── Load satu file hasil training ─────────────────────────────────────────────
model_path = Path(__file__).parent.parent / "3train model" / "hasil_training_svm.pkl"
hasil = joblib.load(model_path)

le          = hasil["label_encoder"]
y_test      = hasil["y_test"]
y_pred      = hasil["y_pred"]
cv_score    = hasil["cv_score"]
acc         = hasil["acc"]
f1          = hasil["f1"]
X_test      = hasil["X_test"]   # ← pastikan disimpan saat training
class_names = le.classes_

base_dir = Path(__file__).parent / "SVM"
base_dir.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1. Confusion Matrix
# ══════════════════════════════════════════════════════════════════════════════
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 1.2),
                                max(5, len(class_names) * 1.0)))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    linewidths=0.5,
    linecolor='#e0e0e0',
    xticklabels=class_names,
    yticklabels=class_names,
    ax=ax,
    cbar_kws={'shrink': 0.75, 'label': 'Jumlah Sampel'},
    annot_kws={'size': 12, 'weight': 'bold'},
)

# Tandai diagonal (prediksi benar) dengan border biru tua
for i in range(len(class_names)):
    ax.add_patch(plt.Rectangle((i, i), 1, 1,
                                fill=False, edgecolor='#0d47a1',
                                lw=2.5, clip_on=False))

ax.set_xlabel('Prediksi',  fontsize=13, labelpad=10, fontweight='bold')
ax.set_ylabel('Aktual',    fontsize=13, labelpad=10, fontweight='bold')
ax.set_title(
    'Confusion Matrix — SVM\n'
    f'Akurasi: {acc:.2%}  |  F1-Score: {f1:.4f}  |  CV Mean: {cv_score.mean():.4f}',
    fontsize=14, fontweight='bold', pad=18
)

ax.tick_params(axis='x', rotation=30, labelsize=10)
ax.tick_params(axis='y', rotation=0,  labelsize=10)

plt.tight_layout()
plt.savefig(base_dir / "confusion_matrix_svm.png", dpi=150, bbox_inches='tight')
plt.close()
print("Confusion matrix disimpan ke: SVM/confusion_matrix_svm.png")

# ══════════════════════════════════════════════════════════════════════════════
# 2. PCA 3D — Penyebaran Data Ekstraksi Uji
# ══════════════════════════════════════════════════════════════════════════════

# Reduksi dimensi ke 3 komponen utama
pca   = PCA(n_components=3, random_state=42)
X_pca = pca.fit_transform(X_test)

# Palet warna per kelas
palette = [
    '#e53935', '#43a047', '#1e88e5', '#fb8c00',
    '#8e24aa', '#00acc1', '#f4511e', '#6d4c41',
]

fig  = plt.figure(figsize=(12, 8))
ax3d = fig.add_subplot(111, projection='3d')

for idx, cls in enumerate(class_names):
    mask = (y_test == idx)
    col  = palette[idx % len(palette)]
    ax3d.scatter(
        X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2],
        c=col, label=cls,
        s=40, alpha=0.75,
        edgecolors='white', linewidths=0.4,
        depthshade=True,
    )

# Tandai titik yang salah diklasifikasi
wrong_mask = (y_test != y_pred)
if wrong_mask.sum() > 0:
    ax3d.scatter(
        X_pca[wrong_mask, 0], X_pca[wrong_mask, 1], X_pca[wrong_mask, 2],
        c='black', marker='x', s=60, linewidths=1.2,
        label='Salah klasifikasi', zorder=5,
    )

# Label sumbu dengan variansi yang dijelaskan
var = pca.explained_variance_ratio_ * 100
ax3d.set_xlabel(f'PC 1 ({var[0]:.1f}%)', fontsize=10, labelpad=8)
ax3d.set_ylabel(f'PC 2 ({var[1]:.1f}%)', fontsize=10, labelpad=8)
ax3d.set_zlabel(f'PC 3 ({var[2]:.1f}%)', fontsize=10, labelpad=8)

ax3d.set_title(
    'Penyebaran Data Ekstraksi Uji — PCA 3D\n'
    f'SVM  |  Total variansi: {sum(var):.1f}%',
    fontsize=13, fontweight='bold', pad=16
)

ax3d.legend(
    fontsize=9, loc='upper left',
    bbox_to_anchor=(0.0, 1.0),
    framealpha=0.7, markerscale=1.4,
)

ax3d.view_init(elev=20, azim=-60)
ax3d.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig(base_dir / "pca_3d_scatter_svm.png", dpi=150, bbox_inches='tight')
plt.close()
print("PCA 3D scatter disimpan ke: SVM/pca_3d_scatter_svm.png")