import os
import cv2
import numpy as np
import pandas as pd
from scipy.stats import skew
from skimage.feature import graycomatrix, graycoprops

dataset_path = "dataset"

color_data = []
texture_data = []
shape_data = []

ANGLES = [0, 45, 90, 135]
ANGLE_LABELS = ["0", "45", "90", "135"]

if not os.path.exists(dataset_path):
    print(f"Error: Folder '{dataset_path}' tidak ditemukan.")
else:
    for label in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, label)

        if not os.path.isdir(folder_path):
            continue

        print("Memproses kelas:", label)

        for file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path)

            if img is None:
                continue

            img = cv2.resize(img, (256, 256))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            r = img[:, :, 2].astype(np.float64)
            g = img[:, :, 1].astype(np.float64)
            b = img[:, :, 0].astype(np.float64)

            color_row = [
                file,
                np.mean(r), np.mean(g), np.mean(b),          # Mean
                np.std(r),  np.std(g),  np.std(b),            # Std Dev
                skew(r.ravel()), skew(g.ravel()), skew(b.ravel()),  # Skewness
                label,
            ]
            color_data.append(color_row)

            angles_rad = [np.deg2rad(a) for a in ANGLES]
            glcm = graycomatrix(
                gray,
                distances=[1],
                angles=angles_rad,
                levels=256,
                symmetric=True,
                normed=True,
            )

            texture_row = [file]
            for i, ang in enumerate(ANGLE_LABELS):
                texture_row += [
                    graycoprops(glcm, 'contrast')[0, i],
                    graycoprops(glcm, 'energy')[0, i],
                    graycoprops(glcm, 'homogeneity')[0, i],
                    graycoprops(glcm, 'correlation')[0, i],
                ]
            texture_row.append(label)
            texture_data.append(texture_row)

            # ── Ekstraksi Fitur Bentuk (Contour) ─────────────────────────────────
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) > 0:
                cnt = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(cnt)
                perimeter = cv2.arcLength(cnt, True)
            else:
                area = 0
                perimeter = 0

            shape_data.append([file, area, perimeter, label])

    color_cols = [
        "file",
        "mean_r", "mean_g", "mean_b",
        "std_r",  "std_g",  "std_b",
        "skew_r", "skew_g", "skew_b",
        "label",
    ]

    texture_cols = ["file"]
    for ang in ANGLE_LABELS:
        texture_cols += [
            f"contrast_{ang}",
            f"energy_{ang}",
            f"homogeneity_{ang}",
            f"correlation_{ang}",
        ]
    texture_cols.append("label")

    shape_cols = ["file", "area", "perimeter", "label"]

    df_color   = pd.DataFrame(color_data,   columns=color_cols)
    df_texture = pd.DataFrame(texture_data, columns=texture_cols)
    df_shape   = pd.DataFrame(shape_data,   columns=shape_cols)

    output_file = "fitur_daun_baru.xlsx"

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        df_color.to_excel(writer,   sheet_name="Fitur Warna",   index=False)
        df_texture.to_excel(writer, sheet_name="Fitur Tekstur", index=False)
        df_shape.to_excel(writer,   sheet_name="Fitur Bentuk",  index=False)

    print("Ekstraksi fitur selesai!")
    print("Data tersimpan di:", output_file)