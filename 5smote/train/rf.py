import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE

# ── Path ───────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR.parent / "fitur_daun_baru.xlsx"
OUT_DIR    = BASE_DIR / "train"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
sheets     = pd.read_excel(INPUT_FILE, sheet_name=None)
df_color   = sheets['Fitur Warna']
df_texture = sheets['Fitur Tekstur']
df_shape   = sheets['Fitur Bentuk']

df_color.columns   = df_color.columns.str.strip()
df_texture.columns = df_texture.columns.str.strip()
df_shape.columns   = df_shape.columns.str.strip()

# ── Definisi fitur ─────────────────────────────────────────────────────────────
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

# ── Gabungkan fitur ────────────────────────────────────────────────────────────
X = np.hstack([
    df_color[color_cols].values,
    df_texture[texture_cols].values,
    df_shape[shape_cols].values,
])
feature_cols = color_cols + texture_cols + shape_cols
y            = df_color['label'].values

# ── Encoding & scaling ─────────────────────────────────────────────────────────
le       = LabelEncoder()
y_enc    = le.fit_transform(y)
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Split data ─────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# ── SMOTE — hanya pada data training ──────────────────────────────────────────
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)
print(f"Jumlah sampel setelah SMOTE: {len(y_train)}")

# ── Training ───────────────────────────────────────────────────────────────────
cv       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred   = rf_model.predict(X_test)
cv_score = cross_val_score(rf_model, X_scaled, y_enc, cv=cv)
acc      = accuracy_score(y_test, y_pred)
f1       = f1_score(y_test, y_pred, average='weighted')

# ── Cetak hasil ────────────────────────────────────────────────────────────────
print("\n===== RANDOM FOREST MODEL =====")
print(f"\nAkurasi : {acc:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"CV Mean : {cv_score.mean():.4f}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))
# print("\nFeature Importance:\n")
# for name, val in zip(feature_cols, rf_model.feature_importances_):
#     print(f"  {name}: {val:.4f}")

# ── Simpan hasil ──────────────────────────────────────────────────────────────
hasil = {
    "model"        : rf_model,
    "label_encoder": le,
    "scaler"       : scaler,
    "feature_cols" : feature_cols,
    "X_test"       : X_test,
    "y_test"       : y_test,
    "y_pred"       : y_pred,
    "cv_score"     : cv_score,
    "acc"          : acc,
    "f1"           : f1,
}
joblib.dump(hasil, OUT_DIR / "hasil_training_rf.pkl")
print(f"\nHasil disimpan ke: {OUT_DIR / 'hasil_training_rf.pkl'}")
