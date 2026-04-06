import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops

# lokasi dataset 
dataset_path = "dataset"

# List untuk menampung data fitur
color_data = []
texture_data = []
shape_data = []

# mengecek apakah folder dataset ada
if not os.path.exists(dataset_path):
    print(f"Error: Folder '{dataset_path}' tidak ditemukan.")
else:
    # membaca setiap folder kelas di dalam dataset nya
    for label in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, label)

        # cek apakah folder
        if not os.path.isdir(folder_path):
            continue

        print("Memproses kelas:", label)

        # membaca setiap file gambar di dalam folder kelas
        for file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path)

            if img is None:
                continue

            # melakukan processing pada gambar, seperti resize dan konversi ke grayscale
            img = cv2.resize(img, (256, 256))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            #Ekstraksi Fitur Warna (Color Moments)
            mean_r = np.mean(img[:, :, 2])
            mean_g = np.mean(img[:, :, 1])
            mean_b = np.mean(img[:, :, 0])
            std_r = np.std(img[:, :, 2])
            std_g = np.std(img[:, :, 1])
            std_b = np.std(img[:, :, 0])

            color_data.append([file, mean_r, mean_g, mean_b, std_r, std_g, std_b, label])

            #Ekstraksi Fitur Tekstur (GLCM)
            glcm = graycomatrix(gray,
                                distances=[1],
                                angles=[0],
                                levels=256,
                                symmetric=True,
                                normed=True)

            contrast = graycoprops(glcm, 'contrast')[0, 0]
            energy = graycoprops(glcm, 'energy')[0, 0]
            homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
            correlation = graycoprops(glcm, 'correlation')[0, 0]

            texture_data.append([file, contrast, energy, homogeneity, correlation, label])

            #Ekstraksi Fitur Bentuk (Contour)
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

    # Membuat DataFrame untuk masing-masing kategori fitur
    df_color = pd.DataFrame(color_data, columns=["file", "mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b", "label"])
    df_texture = pd.DataFrame(texture_data, columns=["file", "contrast", "energy", "homogeneity", "correlation", "label"])
    df_shape = pd.DataFrame(shape_data, columns=["file", "area", "perimeter", "label"])

    # Menyimpan ke satu file Excel dengan sheet yang berbeda
    output_file = "fitur_daun_terpisah.xlsx"

    with pd.ExcelWriter(output_file) as writer:
        df_color.to_excel(writer, sheet_name='Fitur Warna', index=False)
        df_texture.to_excel(writer, sheet_name='Fitur Tekstur', index=False)
        df_shape.to_excel(writer, sheet_name='Fitur Bentuk', index=False)

    print("Ekstraksi fitur selesai!")
    print("Data tersimpan di:", output_file)