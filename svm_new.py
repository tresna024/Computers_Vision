import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)

# load data
input_file = "fitur_daun_baru.xlsx"

sheets     = pd.read_excel(input_file, sheet_name=None)
df_color   = sheets['Fitur Warna']
df_texture = sheets['Fitur Tekstur']
df_shape   = sheets['Fitur Bentuk']

# Bersihkan nama kolom dari spasi tersembunyi
df_color.columns   = df_color.columns.str.strip()
df_texture.columns = df_texture.columns.str.strip()
df_shape.columns   = df_shape.columns.str.strip()

# fitur yang digunakan untuk model
color_cols = [
    'mean_r', 'mean_g', 'mean_b',
    'std_r',  'std_g',  'std_b',
    'skew_r', 'skew_g', 'skew_b'
]

texture_cols = [
    'contrast_0',   'energy_0',   'homogeneity_0',   'correlation_0',
    'contrast_45',  'energy_45',  'homogeneity_45',  'correlation_45',
    'contrast_90',  'energy_90',  'homogeneity_90',  'correlation_90',
    'contrast_135', 'energy_135', 'homogeneity_135', 'correlation_135',
]

shape_cols = ['area', 'perimeter']

X_color   = df_color[color_cols].values
X_texture = df_texture[texture_cols].values
X_shape   = df_shape[shape_cols].values

# Gabungkan secara horizontal hanya sebagai input model
X = np.hstack([X_color, X_texture, X_shape])
feature_cols = color_cols + texture_cols + shape_cols

y = df_color['label'].values  # label diambil dari sheet warna

# encoding label
le = LabelEncoder()
y_enc = le.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n===== SVM MODEL =====")

svm_model = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
svm_model.fit(X_train, y_train)

y_pred   = svm_model.predict(X_test)
cv_score = cross_val_score(svm_model, X_scaled, y_enc, cv=cv)

acc = accuracy_score(y_test, y_pred)
f1  = f1_score(y_test, y_pred, average='weighted')

print(f"\nAkurasi : {acc:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"CV Mean : {cv_score.mean():.4f}")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# simpan gambar
cm          = confusion_matrix(y_test, y_pred)
class_names = le.classes_

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
plt.savefig("confusion_matrix_svm.png", dpi=150, bbox_inches='tight')
plt.close()
print("\nConfusion matrix disimpan ke: confusion_matrix_svm.png")