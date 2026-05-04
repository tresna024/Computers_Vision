import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
from pathlib import Path

from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ── Load satu file hasil training ─────────────────────────────────────────────
model_path = Path(__file__).parent.parent / "3train model" / "hasil_training_rf.pkl"
hasil = joblib.load(model_path)

rf_model     = hasil["model"]
le           = hasil["label_encoder"]
feature_cols = hasil["feature_cols"]
y_test       = hasil["y_test"]
y_pred       = hasil["y_pred"]
cv_score     = hasil["cv_score"]
acc          = hasil["acc"]
f1           = hasil["f1"]
class_names  = le.classes_

# Ambil data fitur uji dari hasil training (pastikan X_test tersimpan)
# Jika X_test tidak ada di pkl, gunakan: X_test = hasil["X_test"]
X_test = hasil["X_test"]  # tambahkan ini saat menyimpan model

base_dir = Path(__file__).parent / "Random Forest"
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
    cmap='YlGn',
    linewidths=0.5,
    linecolor='#e0e0e0',
    xticklabels=class_names,
    yticklabels=class_names,
    ax=ax,
    cbar_kws={'shrink': 0.75, 'label': 'Jumlah Sampel'},
    annot_kws={'size': 12, 'weight': 'bold'},
)

for i in range(len(class_names)):
    ax.add_patch(plt.Rectangle((i, i), 1, 1,
                                fill=False, edgecolor='#2e7d32',
                                lw=2.5, clip_on=False))

ax.set_xlabel('Prediksi',  fontsize=13, labelpad=10, fontweight='bold')
ax.set_ylabel('Aktual',    fontsize=13, labelpad=10, fontweight='bold')
ax.set_title(
    'Confusion Matrix — Random Forest\n'
    f'Akurasi: {acc:.2%}  |  F1-Score: {f1:.4f}  |  CV Mean: {cv_score.mean():.4f}',
    fontsize=14, fontweight='bold', pad=18
)

ax.tick_params(axis='x', rotation=30, labelsize=10)
ax.tick_params(axis='y', rotation=0,  labelsize=10)

plt.tight_layout()
plt.savefig(base_dir / "confusion_matrix_rf.png", dpi=150, bbox_inches='tight')
plt.close()
print("Confusion matrix disimpan ke: Random Forest/confusion_matrix_rf.png")

# ══════════════════════════════════════════════════════════════════════════════
# 2. Feature Importance (Top 20)
# ══════════════════════════════════════════════════════════════════════════════
color_cols = ['mean_r','mean_g','mean_b','std_r','std_g','std_b','skew_r','skew_g','skew_b']
shape_cols = ['area', 'perimeter']

def feat_color(name):
    if name in color_cols:   return '#1565c0'
    elif name in shape_cols: return '#e65100'
    else:                    return '#2e7d32'

importances = rf_model.feature_importances_
indices     = np.argsort(importances)[::-1][:20]
top_names   = [feature_cols[i] for i in indices]
top_vals    = importances[indices]
colors      = [feat_color(n) for n in top_names]

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(top_names[::-1], top_vals[::-1], color=colors[::-1], edgecolor='white')
ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
ax.set_title('Feature Importance — Top 20\nRandom Forest', fontsize=14, fontweight='bold', pad=14)
ax.tick_params(axis='y', labelsize=9)
ax.tick_params(axis='x', labelsize=10)
ax.xaxis.grid(True, linestyle='--', alpha=0.6)
ax.set_axisbelow(True)

legend_patches = [
    mpatches.Patch(color='#1565c0', label='Warna'),
    mpatches.Patch(color='#2e7d32', label='Tekstur'),
    mpatches.Patch(color='#e65100', label='Bentuk'),
]
ax.legend(handles=legend_patches, fontsize=10, loc='lower right')

plt.tight_layout()
plt.savefig(base_dir / "feature_importance_rf.png", dpi=150, bbox_inches='tight')
plt.close()
print("Feature importance disimpan ke: Random Forest/feature_importance_rf.png")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
from pathlib import Path

from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ── Load satu file hasil training ─────────────────────────────────────────────
model_path = Path(__file__).parent.parent / "3train model" / "hasil_training_rf.pkl"
hasil = joblib.load(model_path)

rf_model     = hasil["model"]
le           = hasil["label_encoder"]
feature_cols = hasil["feature_cols"]
y_test       = hasil["y_test"]
y_pred       = hasil["y_pred"]
cv_score     = hasil["cv_score"]
acc          = hasil["acc"]
f1           = hasil["f1"]
class_names  = le.classes_

# Ambil data fitur uji dari hasil training (pastikan X_test tersimpan)
# Jika X_test tidak ada di pkl, gunakan: X_test = hasil["X_test"]
X_test = hasil["X_test"]  # tambahkan ini saat menyimpan model

base_dir = Path(__file__).parent / "Random Forest"
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
    cmap='YlGn',
    linewidths=0.5,
    linecolor='#e0e0e0',
    xticklabels=class_names,
    yticklabels=class_names,
    ax=ax,
    cbar_kws={'shrink': 0.75, 'label': 'Jumlah Sampel'},
    annot_kws={'size': 12, 'weight': 'bold'},
)

for i in range(len(class_names)):
    ax.add_patch(plt.Rectangle((i, i), 1, 1,
                                fill=False, edgecolor='#2e7d32',
                                lw=2.5, clip_on=False))

ax.set_xlabel('Prediksi',  fontsize=13, labelpad=10, fontweight='bold')
ax.set_ylabel('Aktual',    fontsize=13, labelpad=10, fontweight='bold')
ax.set_title(
    'Confusion Matrix — Random Forest\n'
    f'Akurasi: {acc:.2%}  |  F1-Score: {f1:.4f}  |  CV Mean: {cv_score.mean():.4f}',
    fontsize=14, fontweight='bold', pad=18
)

ax.tick_params(axis='x', rotation=30, labelsize=10)
ax.tick_params(axis='y', rotation=0,  labelsize=10)

plt.tight_layout()
plt.savefig(base_dir / "confusion_matrix_rf.png", dpi=150, bbox_inches='tight')
plt.close()
print("Confusion matrix disimpan ke: Random Forest/confusion_matrix_rf.png")

# ══════════════════════════════════════════════════════════════════════════════
# 2. Feature Importance (Top 20)
# ══════════════════════════════════════════════════════════════════════════════
color_cols = ['mean_r','mean_g','mean_b','std_r','std_g','std_b','skew_r','skew_g','skew_b']
shape_cols = ['area', 'perimeter']

def feat_color(name):
    if name in color_cols:   return '#1565c0'
    elif name in shape_cols: return '#e65100'
    else:                    return '#2e7d32'

importances = rf_model.feature_importances_
indices     = np.argsort(importances)[::-1][:20]
top_names   = [feature_cols[i] for i in indices]
top_vals    = importances[indices]
colors      = [feat_color(n) for n in top_names]

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(top_names[::-1], top_vals[::-1], color=colors[::-1], edgecolor='white')
ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
ax.set_title('Feature Importance — Top 20\nRandom Forest', fontsize=14, fontweight='bold', pad=14)
ax.tick_params(axis='y', labelsize=9)
ax.tick_params(axis='x', labelsize=10)
ax.xaxis.grid(True, linestyle='--', alpha=0.6)
ax.set_axisbelow(True)

legend_patches = [
    mpatches.Patch(color='#1565c0', label='Warna'),
    mpatches.Patch(color='#2e7d32', label='Tekstur'),
    mpatches.Patch(color='#e65100', label='Bentuk'),
]
ax.legend(handles=legend_patches, fontsize=10, loc='lower right')

plt.tight_layout()
plt.savefig(base_dir / "feature_importance_rf.png", dpi=150, bbox_inches='tight')
plt.close()
print("Feature importance disimpan ke: Random Forest/feature_importance_rf.png")

# ══════════════════════════════════════════════════════════════════════════════
# 3. PCA 3D — Penyebaran Data Ekstraksi Uji
# ══════════════════════════════════════════════════════════════════════════════

# Reduksi dimensi ke 3 komponen utama
pca = PCA(n_components=3, random_state=42)
X_pca = pca.fit_transform(X_test)

# Warna per kelas — sesuaikan jika jumlah kelas berbeda
palette = [
    '#e53935',  # merah
    '#43a047',  # hijau
    '#1e88e5',  # biru
    '#fb8c00',  # oranye
    '#8e24aa',  # ungu
    '#00acc1',  # cyan
    '#f4511e',  # merah-oranye
    '#6d4c41',  # coklat
]

fig = plt.figure(figsize=(12, 8))
ax3d = fig.add_subplot(111, projection='3d')

for idx, cls in enumerate(class_names):
    mask = (y_test == idx)          # y_test dalam bentuk label encoded (int)
    col  = palette[idx % len(palette)]
    ax3d.scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        X_pca[mask, 2],
        c=col,
        label=cls,
        s=40,
        alpha=0.75,
        edgecolors='white',
        linewidths=0.4,
        depthshade=True,
    )

# Tampilkan juga titik prediksi yang salah (opsional — tandai dengan 'x')
wrong_mask = (y_test != y_pred)
if wrong_mask.sum() > 0:
    ax3d.scatter(
        X_pca[wrong_mask, 0],
        X_pca[wrong_mask, 1],
        X_pca[wrong_mask, 2],
        c='black',
        marker='x',
        s=60,
        linewidths=1.2,
        label='Salah klasifikasi',
        zorder=5,
    )

# Label sumbu dengan variansi yang dijelaskan
var = pca.explained_variance_ratio_ * 100
ax3d.set_xlabel(f'PC 1 ({var[0]:.1f}%)', fontsize=10, labelpad=8)
ax3d.set_ylabel(f'PC 2 ({var[1]:.1f}%)', fontsize=10, labelpad=8)
ax3d.set_zlabel(f'PC 3 ({var[2]:.1f}%)', fontsize=10, labelpad=8)

ax3d.set_title(
    'Penyebaran Data Ekstraksi Uji — PCA 3D\n'
    f'Random Forest  |  Total variansi: {sum(var):.1f}%',
    fontsize=13, fontweight='bold', pad=16
)

ax3d.legend(
    fontsize=9,
    loc='upper left',
    bbox_to_anchor=(0.0, 1.0),
    framealpha=0.7,
    markerscale=1.4,
)

ax3d.view_init(elev=20, azim=-60)   # sudut pandang default; bisa diubah
ax3d.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig(base_dir / "pca_3d_scatter_rf.png", dpi=150, bbox_inches='tight')
plt.close()
print("PCA 3D scatter disimpan ke: Random Forest/pca_3d_scatter_rf.png")