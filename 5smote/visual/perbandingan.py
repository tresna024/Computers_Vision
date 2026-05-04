import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

# ── Base Path ─────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent   # visual
ROOT_DIR = BASE_DIR.parent                   # 5smote

# ── Path ──────────────────────────────────────────────
HASIL_DIR = ROOT_DIR / "train"
OUT_DIR   = BASE_DIR / "hasil Perbandingan"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load model ────────────────────────────────────────
hasil_rf  = joblib.load(HASIL_DIR / "hasil_training_rf.pkl")
hasil_svm = joblib.load(HASIL_DIR / "hasil_training_svm.pkl")
hasil_xgb = joblib.load(HASIL_DIR / "hasil_training_xgb.pkl")

names    = ["Random Forest", "SVM", "XGBoost"]
acc_vals = [hasil_rf["acc"], hasil_svm["acc"], hasil_xgb["acc"]]
colors   = ["#2e7d32", "#1565c0", "#e65100"]

# ── Diagram Batang Akurasi ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))

bars = ax.bar(names, acc_vals, color=colors, width=0.5, edgecolor='white', zorder=3)

for bar, val in zip(bars, acc_vals):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.4f}",
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel('Akurasi', fontsize=12, fontweight='bold')
ax.set_title('Perbandingan Akurasi Model\nRandom Forest vs SVM vs XGBoost (WITH SMOTE)',
             fontsize=14, fontweight='bold', pad=16)
ax.set_ylim(0, 1.12)
ax.yaxis.grid(True, linestyle='--', alpha=0.5)
ax.set_axisbelow(True)
ax.tick_params(axis='x', labelsize=11)

plt.tight_layout()
plt.savefig(OUT_DIR / "perbandingan_akurasi.png", dpi=150, bbox_inches='tight')
plt.close()
print("Diagram perbandingan disimpan ke: hasil/perbandingan_akurasi.png")

# ── Ringkasan terminal ─────────────────────────────────────────────────────────
cv_means = [hasil_rf["cv_score"].mean(), hasil_svm["cv_score"].mean(), hasil_xgb["cv_score"].mean()]
f1_vals  = [hasil_rf["f1"], hasil_svm["f1"], hasil_xgb["f1"]]

print("\n" + "="*52)
print(f"{'Model':<16} {'Akurasi':>9} {'F1-Score':>9} {'CV Mean':>9}")
print("="*52)
for name, acc, f1, cv_m in zip(names, acc_vals, f1_vals, cv_means):
    print(f"{name:<16} {acc:>9.4f} {f1:>9.4f} {cv_m:>9.4f}")
print("="*52)
best = names[np.argmax(acc_vals)]
print(f"\nModel terbaik (Akurasi): {best} ({max(acc_vals):.4f})")
