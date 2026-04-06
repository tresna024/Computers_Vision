import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# LOAD DATA
input_file = "fitur_daun_terpisah.xlsx"

sheets     = pd.read_excel(input_file, sheet_name=None)
df_color   = sheets['Fitur Warna']
df_texture = sheets['Fitur Tekstur']
df_shape   = sheets['Fitur Bentuk']

df_all = (df_color
          .merge(df_texture.drop(columns=['label']), on='file')
          .merge(df_shape.drop(columns=['label']),   on='file'))

# FIX kolom (biar aman dari error sebelumnya)
df_all.columns = df_all.columns.str.strip()

feature_cols = ['mean_r','mean_g','mean_b','std_r','std_g','std_b',
                'contrast','energy','homogeneity','correlation',
                'area','perimeter']

X = df_all[feature_cols].values
y = df_all['label'].values

le = LabelEncoder()
y_enc = le.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# MODEL SVM
print("\n===== SVM MODEL =====")

svm_model = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)

print("\nAkurasi:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))

cv_score = cross_val_score(svm_model, X_scaled, y_enc, cv=cv)
print("CV Mean:", cv_score.mean())

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))