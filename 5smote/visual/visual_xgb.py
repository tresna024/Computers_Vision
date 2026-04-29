import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
from pathlib import Path

from sklearn.metrics import confusion_matrix

# ── Path ───────────────────────────────────────────────────────────────────────
# 1. Tentukan root project (5smote)
BASE_DIR = Path(__file__).resolve().parent   # → .../5smote/visual
ROOT_DIR = BASE_DIR.parent                   # → .../5smote

# 2. Arahkan ke folder train yang benar
HASIL_DIR = ROOT_DIR / "train"               # → .../5smote/train
OUT_DIR = BASE_DIR / "hasil"               # → .../5smote/train

# 3. Load model
hasil = joblib.load(HASIL_DIR / "hasil_training_xgb.pkl")

xgb_model    = hasil["model"]
le           = hasil["label_encoder"]
feature_cols = hasil["feature_cols"]
y_test       = hasil["y_test"]
y_pred       = hasil["y_pred"]
cv_score     = hasil["cv_score"]
acc          = hasil["acc"]
f1           = hasil["f1"]
class_names  = le.classes_

# ══════════════════════════════════════════════════════════════════════════════
# 1. Confusion Matrix
# ══════════════════════════════════════════════════════════════════════════════
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 1.2),
                                max(5, len(class_names) * 1.0)))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Oranges',
    linewidths=0.5, linecolor='#e0e0e0',
    xticklabels=class_names, yticklabels=class_names, ax=ax,
    cbar_kws={'shrink': 0.75, 'label': 'Jumlah Sampel'},
    annot_kws={'size': 12, 'weight': 'bold'},
)
for i in range(len(class_names)):
    ax.add_patch(plt.Rectangle((i, i), 1, 1,
                                fill=False, edgecolor='#e65100', lw=2.5, clip_on=False))
ax.set_xlabel('Prediksi',  fontsize=13, labelpad=10, fontweight='bold')
ax.set_ylabel('Aktual',    fontsize=13, labelpad=10, fontweight='bold')
ax.set_title(
    'Confusion Matrix — XGBoost\n'
    f'Akurasi: {acc:.2%}  |  F1-Score: {f1:.4f}  |  CV Mean: {cv_score.mean():.4f}',
    fontsize=14, fontweight='bold', pad=18
)
ax.tick_params(axis='x', rotation=30, labelsize=10)
ax.tick_params(axis='y', rotation=0,  labelsize=10)
plt.tight_layout()
plt.savefig(OUT_DIR / "confusion_matrix_xgb.png", dpi=150, bbox_inches='tight')
plt.close()
print("Confusion matrix disimpan ke: hasil/confusion_matrix_xgb.png")

# ══════════════════════════════════════════════════════════════════════════════
# 2. Feature Importance (Top 20)
# ══════════════════════════════════════════════════════════════════════════════
color_cols = ['mean_r','mean_g','mean_b','std_r','std_g','std_b','skew_r','skew_g','skew_b']
shape_cols = ['area', 'perimeter']

def feat_color(name):
    if name in color_cols:   return '#1565c0'
    elif name in shape_cols: return '#e65100'
    else:                    return '#2e7d32'

importances = xgb_model.feature_importances_
indices     = np.argsort(importances)[::-1][:20]
top_names   = [feature_cols[i] for i in indices]
top_vals    = importances[indices]
colors      = [feat_color(n) for n in top_names]

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(top_names[::-1], top_vals[::-1], color=colors[::-1], edgecolor='white')
ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
ax.set_title('Feature Importance — Top 20\nXGBoost', fontsize=14, fontweight='bold', pad=14)
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
plt.savefig(OUT_DIR / "feature_importance_xgb.png", dpi=150, bbox_inches='tight')
plt.close()
print("Feature importance disimpan ke: hasil/feature_importance_xgb.png")
