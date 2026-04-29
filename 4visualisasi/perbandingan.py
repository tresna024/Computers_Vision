import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
from pathlib import Path

# ── Load hasil ketiga model ────────────────────────────────────────────────────
model_path_rf = Path(__file__).parent.parent / "3train model" / "hasil_training_rf.pkl"
model_path_svm = Path(__file__).parent.parent / "3train model" / "hasil_training_svm.pkl"
model_path_xgb = Path(__file__).parent.parent / "3train model" / "hasil_training_xgboost.pkl"

hasil_rf  = joblib.load(model_path_rf)
hasil_svm = joblib.load(model_path_svm)
hasil_xgb = joblib.load(model_path_xgb)

names    = ["Random Forest", "SVM", "XGBoost"]
acc_vals = [hasil_rf["acc"], hasil_svm["acc"], hasil_xgb["acc"]]
colors   = ["#2e7d32", "#1565c0", "#e65100"]

base_dir = Path(__file__).parent
 
# ── Diagram Batang Akurasi ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
 
bars = ax.bar(names, acc_vals, color=colors, width=0.5, edgecolor='white', zorder=3)
 
for bar, val in zip(bars, acc_vals):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.4f}",
            ha='center', va='bottom', fontsize=11, fontweight='bold')
 
ax.set_ylabel('Akurasi', fontsize=12, fontweight='bold')
ax.set_title('Perbandingan Akurasi Model\nRandom Forest vs SVM vs XGBoost',
             fontsize=14, fontweight='bold', pad=16)
ax.set_ylim(0, 1.12)
ax.yaxis.grid(True, linestyle='--', alpha=0.5)
ax.set_axisbelow(True)
ax.tick_params(axis='x', labelsize=11)
 
plt.tight_layout()
plt.savefig(base_dir / "perbandingan.png", dpi=150, bbox_inches='tight')
plt.close()
print("Diagram batang akurasi disimpan ke: perbandingan.png")