import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

from sklearn.metrics import confusion_matrix

# ── Load satu file hasil training ─────────────────────────────────────────────
model_path = Path(__file__).parent.parent / "3train model" / "hasil_training_svm.pkl"
hasil = joblib.load(model_path)

le          = hasil["label_encoder"]
y_test      = hasil["y_test"]
y_pred      = hasil["y_pred"]
cv_score    = hasil["cv_score"]
acc         = hasil["acc"]
f1          = hasil["f1"]
class_names = le.classes_

base_dir = Path(__file__).parent / "SVM"
# ══════════════════════════════════════════════════════════════════════════════
# Confusion Matrix
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
print("Confusion matrix disimpan ke: confusion_matrix_svm.png")