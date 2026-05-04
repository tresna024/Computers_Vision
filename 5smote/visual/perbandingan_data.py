import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

# ── Path ───────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR.parent / "fitur_daun_baru.xlsx"
HASIL_DIR  = Path(__file__).resolve().parent / "hasil Perbandingan"
HASIL_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
sheets     = pd.read_excel(INPUT_FILE, sheet_name=None)
df_color   = sheets['Fitur Warna']
df_texture = sheets['Fitur Tekstur']
df_shape   = sheets['Fitur Bentuk']

df_color.columns   = df_color.columns.str.strip()
df_texture.columns = df_texture.columns.str.strip()
df_shape.columns   = df_shape.columns.str.strip()

# ── Gabungkan fitur ────────────────────────────────────────────────────────────
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

X = np.hstack([
    df_color[color_cols].values,
    df_texture[texture_cols].values,
    df_shape[shape_cols].values,
])
y = df_color['label'].values

# ── Encoding & scaling ─────────────────────────────────────────────────────────
le       = LabelEncoder()
y_enc    = le.fit_transform(y)
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Split data ─────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# ── Hitung distribusi sebelum & setelah SMOTE ─────────────────────────────────
counter_before         = Counter(y_train)
X_train_sm, y_train_sm = SMOTE(random_state=42).fit_resample(X_train, y_train)
counter_after          = Counter(y_train_sm)

# ── Susun data untuk plot ──────────────────────────────────────────────────────
class_names   = le.classes_
before_counts = [counter_before.get(i, 0) for i in range(len(class_names))]
after_counts  = [counter_after.get(i, 0)  for i in range(len(class_names))]

# ── Cetak ringkasan terminal ───────────────────────────────────────────────────
print("=" * 52)
print(f"{'Kelas':<18} {'Sebelum SMOTE':>14} {'Setelah SMOTE':>14}")
print("=" * 52)
for name, b, a in zip(class_names, before_counts, after_counts):
    print(f"{name:<18} {b:>14} {a:>14}")
print("=" * 52)
print(f"{'TOTAL':<18} {sum(before_counts):>14} {sum(after_counts):>14}")
print("=" * 52)

# ══════════════════════════════════════════════════════════════════════════════
# Diagram Batang Perbandingan
# ══════════════════════════════════════════════════════════════════════════════
x     = np.arange(len(class_names))
width = 0.35

fig, ax = plt.subplots(figsize=(max(8, len(class_names) * 0.9), 6))

bars_before = ax.bar(x - width / 2, before_counts, width,
                     label='Sebelum SMOTE', color='#1565c0',
                     edgecolor='white', zorder=3)
bars_after  = ax.bar(x + width / 2, after_counts,  width,
                     label='Setelah SMOTE',  color='#2e7d32',
                     edgecolor='white', zorder=3)

for bar, val in zip(bars_before, before_counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            str(val), ha='center', va='bottom', fontsize=9,
            fontweight='bold', color='#1565c0')

for bar, val in zip(bars_after, after_counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            str(val), ha='center', va='bottom', fontsize=9,
            fontweight='bold', color='#2e7d32')

ax.set_xticks(x)
ax.set_xticklabels(class_names, rotation=30, ha='right', fontsize=10)
ax.set_ylabel('Jumlah Sampel (Data Training)', fontsize=12, fontweight='bold')
ax.set_title(
    'Perbandingan Distribusi Data Training\nSebelum dan Setelah SMOTE',
    fontsize=14, fontweight='bold', pad=16
)
ax.yaxis.grid(True, linestyle='--', alpha=0.5)
ax.set_axisbelow(True)
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig(HASIL_DIR / "perbandingan_data.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"\nDiagram disimpan ke: hasil/perbandingan_data.png")
