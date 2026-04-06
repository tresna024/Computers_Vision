import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
input_file = "fitur_daun_terpisah.xlsx"

sheets     = pd.read_excel(input_file, sheet_name=None)
df_color   = sheets['Fitur Warna']
df_texture = sheets['Fitur Tekstur']
df_shape   = sheets['Fitur Bentuk']

df_all = (df_color
          .merge(df_texture.drop(columns=['label']), on='file')
          .merge(df_shape.drop(columns=['label']),   on='file'))

feature_cols = ['mean_r','mean_g','mean_b','std_r','std_g','std_b',
                'contrast','energy','homogeneity','correlation',
                'area','perimeter']

X = df_all[feature_cols].values
y = df_all['label'].values

le       = LabelEncoder()
y_enc    = le.fit_transform(y)
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ─────────────────────────────────────────────
# 2. HELPER PRINT
# ─────────────────────────────────────────────
SEP  = "=" * 65
SEP2 = "-" * 65

def print_section(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)

def print_confusion_matrix(cm, classes):
    col_w = max(len(c) for c in classes) + 2
    header = " " * col_w + "".join(f"{c:>{col_w}}" for c in classes)
    print(header)
    print(" " * col_w + "-" * (col_w * len(classes)))
    for i, row_label in enumerate(classes):
        row = f"{row_label:<{col_w}}" + "".join(f"{v:>{col_w}}" for v in cm[i])
        print(row)

# ─────────────────────────────────────────────
# 3. SVM
# ─────────────────────────────────────────────
print_section("MODEL 1 : SUPPORT VECTOR MACHINE (SVM - RBF Kernel)")

svm_model = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
svm_model.fit(X_train, y_train)

y_pred_svm = svm_model.predict(X_test)
svm_acc    = accuracy_score(y_test, y_pred_svm)
svm_f1     = f1_score(y_test, y_pred_svm, average='weighted')
svm_cv     = cross_val_score(svm_model, X_scaled, y_enc, cv=cv, scoring='accuracy')

print(f"\n  Konfigurasi  : kernel=rbf | C=10 | gamma=scale")
print(f"  Train size   : {len(X_train)} sampel")
print(f"  Test size    : {len(X_test)} sampel")
print(f"\n  {'─'*35}")
print(f"  Akurasi Test Set : {svm_acc:.4f}  ({svm_acc*100:.2f}%)")
print(f"  F1-Score (W avg) : {svm_f1:.4f}")
print(f"  CV (5-Fold) Mean : {svm_cv.mean():.4f} +/- {svm_cv.std():.4f}")
print(f"  CV Scores        : {[f'{s:.4f}' for s in svm_cv]}")
print(f"  {'─'*35}")

print("\n  CLASSIFICATION REPORT:\n")
print(classification_report(y_test, y_pred_svm, target_names=le.classes_))

print("  CONFUSION MATRIX:\n")
svm_cm = confusion_matrix(y_test, y_pred_svm)
print_confusion_matrix(svm_cm, le.classes_)

# ─────────────────────────────────────────────
# 4. RANDOM FOREST
# ─────────────────────────────────────────────
print_section("MODEL 2 : RANDOM FOREST")

rf_model = RandomForestClassifier(n_estimators=100, max_depth=None,
                                   random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

y_pred_rf  = rf_model.predict(X_test)
rf_acc     = accuracy_score(y_test, y_pred_rf)
rf_f1      = f1_score(y_test, y_pred_rf, average='weighted')
rf_cv      = cross_val_score(rf_model, X_scaled, y_enc, cv=cv, scoring='accuracy')

print(f"\n  Konfigurasi  : n_estimators=100 | max_depth=None")
print(f"  Train size   : {len(X_train)} sampel")
print(f"  Test size    : {len(X_test)} sampel")
print(f"\n  {'─'*35}")
print(f"  Akurasi Test Set : {rf_acc:.4f}  ({rf_acc*100:.2f}%)")
print(f"  F1-Score (W avg) : {rf_f1:.4f}")
print(f"  CV (5-Fold) Mean : {rf_cv.mean():.4f} +/- {rf_cv.std():.4f}")
print(f"  CV Scores        : {[f'{s:.4f}' for s in rf_cv]}")
print(f"  {'─'*35}")

print("\n  CLASSIFICATION REPORT:\n")
print(classification_report(y_test, y_pred_rf, target_names=le.classes_))

print("  CONFUSION MATRIX:\n")
rf_cm = confusion_matrix(y_test, y_pred_rf)
print_confusion_matrix(rf_cm, le.classes_)

# Feature Importance
print("\n  FEATURE IMPORTANCE:\n")
importances = rf_model.feature_importances_
idx_sorted  = np.argsort(importances)[::-1]
print(f"  {'Rank':<6} {'Fitur':<15} {'Importance':>12}  Bar")
print(f"  {'─'*4}  {'─'*13}  {'─'*10}  {'─'*25}")
for rank, i in enumerate(idx_sorted, 1):
    bar = "#" * int(importances[i] * 100)
    print(f"  {rank:<6} {feature_cols[i]:<15} {importances[i]:>12.6f}  {bar}")

# ─────────────────────────────────────────────
# 5. PERBANDINGAN AKHIR
# ─────────────────────────────────────────────
print_section("PERBANDINGAN KEDUA MODEL")

best = "SVM" if svm_acc >= rf_acc else "Random Forest"
print(f"\n  {'Model':<20} {'Akurasi':>10} {'F1-Score':>10} {'CV Mean':>10} {'CV Std':>10}")
print(f"  {'─'*18}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")
svm_marker = " <== TERBAIK" if best == "SVM" else ""
rf_marker  = " <== TERBAIK" if best == "Random Forest" else ""
print(f"  {'SVM (RBF)':<20} {svm_acc:>10.4f} {svm_f1:>10.4f} {svm_cv.mean():>10.4f} {svm_cv.std():>10.4f}{svm_marker}")
print(f"  {'Random Forest':<20} {rf_acc:>10.4f} {rf_f1:>10.4f} {rf_cv.mean():>10.4f} {rf_cv.std():>10.4f}{rf_marker}")

diff = abs(svm_acc - rf_acc) * 100
print(f"\n  Model terbaik (akurasi test set) : {best}")
print(f"  Selisih akurasi                  : {diff:.2f}%")
print(f"\n{SEP}\n")