import matplotlib.pyplot as plt
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
hasil = joblib.load(HASIL_DIR / "hasil_training_svm.pkl")

le          = hasil["label_encoder"]
y_test      = hasil["y_test"]
y_pred      = hasil["y_pred"]
cv_score    = hasil["cv_score"]
acc         = hasil["acc"]
f1          = hasil["f1"]
class_names = le.classes_

# ══════════════════════════════════════════════════════════════════════════════
# Confusion Matrix
# ══════════════════════════════════════════════════════════════════════════════
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 1.2),
                                max(5, len(class_names) * 1.0)))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    linewidths=0.5, linecolor='#e0e0e0',
    xticklabels=class_names, yticklabels=class_names, ax=ax,
    cbar_kws={'shrink': 0.75, 'label': 'Jumlah Sampel'},
    annot_kws={'size': 12, 'weight': 'bold'},
)
for i in range(len(class_names)):
    ax.add_patch(plt.Rectangle((i, i), 1, 1,
                                fill=False, edgecolor='#0d47a1', lw=2.5, clip_on=False))
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
plt.savefig(OUT_DIR / "confusion_matrix_svm.png", dpi=150, bbox_inches='tight')
plt.close()
print("Confusion matrix disimpan ke: hasil/confusion_matrix_svm.png")
