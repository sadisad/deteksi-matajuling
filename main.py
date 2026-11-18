import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import sys

from screening import INCONCLUSIVE, NORMAL, STRABISMUS, Detection, decide, format_result

os.environ['YOLO_VERBOSE'] = 'False'

# --- CONFIG PATH (FIXED) ---
# .resolve() memastikan path selalu absolut, bukan relative
BASE_DIR = Path(__file__).resolve().parent

MODEL_DETEKSI_PATH = BASE_DIR / "models" / "model_mata.pt"
MODEL_KLASIFIKASI_PATH = BASE_DIR / "models" / "model_juling.pt"

# Debug: tampilkan path yang dicari
print(f"📁 BASE_DIR   : {BASE_DIR}")
print(f"📄 Model Mata : {MODEL_DETEKSI_PATH}")
print(f"📄 Model Juling: {MODEL_KLASIFIKASI_PATH}")

# Verifikasi keberadaan model
if not MODEL_DETEKSI_PATH.exists() or not MODEL_KLASIFIKASI_PATH.exists():
    print("\n❌ ERROR: Salah satu model tidak ditemukan!")
    print(f"   model_mata.pt   → {'✅ ADA' if MODEL_DETEKSI_PATH.exists() else '❌ TIDAK ADA'}")
    print(f"   model_juling.pt → {'✅ ADA' if MODEL_KLASIFIKASI_PATH.exists() else '❌ TIDAK ADA'}")
    sys.exit(1)

# Load Model
try:
    detector = YOLO(str(MODEL_DETEKSI_PATH))
    classifier = YOLO(str(MODEL_KLASIFIKASI_PATH))
    print("\n✅ Kedua Model (Deteksi & Klasifikasi) Berhasil Dimuat.")
except Exception as e:
    print(f"❌ Gagal Memuat Model: {e}")
    sys.exit(1)


def process_screening():
    root = tk.Tk()
    root.withdraw()
    img_path = filedialog.askopenfilename(
        title="Pilih Foto Mata",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
    )
    root.destroy()

    if not img_path:
        return

    img = cv2.imread(img_path)
    if img is None:
        messagebox.showerror("Error", f"Gagal membaca gambar:\n{img_path}")
        return

    img_display = img.copy()

    # --- TAHAP 1: DETEKSI AREA MATA (Menggunakan model_mata.pt) ---
    results_det = detector.predict(img, conf=0.15, verbose=False, device='cpu')
    regions = []

    for r in results_det:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            regions.append([x1, y1, x2, y2])

    # Fallback: Jika model_mata.pt gagal, gunakan OpenCV Haar Cascade
    if not regions:
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in eyes:
            # Berdasarkan data training, kotaknya sangat pas dengan mata.
            # Jadi kita hilangkan padding sepenuhnya (pad = 0)
            pad = 0
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(img.shape[1], x + w + pad)
            y2 = min(img.shape[0], y + h + pad)
            regions.append([x1, y1, x2, y2])

    eye_regions_found = bool(regions)

    if not regions:
        # Tidak ada mata terdeteksi. Stage 2 tetap dijalankan pada gambar
        # penuh, tetapi hasilnya dilaporkan INCONCLUSIVE, bukan NORMAL.
        regions.append([0, 0, img.shape[1], img.shape[0]])

    list_mata_terdeteksi = []
    detections = []

    if regions:
        # Buat satu area besar (Horizontal Strip) yang mencakup kedua mata
        # Ini menyesuaikan dengan bentuk data latih model_juling.pt
        min_x = min([r[0] for r in regions])
        min_y = min([r[1] for r in regions])
        max_x = max([r[2] for r in regions])
        max_y = max([r[3] for r in regions])
        
        # Tambahkan padding agar mencakup area sekitar mata (seperti di data latih)
        pad_x, pad_y = 20, 20
        strip_x1 = max(0, min_x - pad_x)
        strip_y1 = max(0, min_y - pad_y)
        strip_x2 = min(img.shape[1], max_x + pad_x)
        strip_y2 = min(img.shape[0], max_y + pad_y)
        
        strip_img = img[strip_y1:strip_y2, strip_x1:strip_x2]
        
        # --- TAHAP 2: KLASIFIKASI PADA STRIP MATA ---
        results_cls = classifier.predict(strip_img, conf=0.5, verbose=False, device='cpu')
        
        for box in results_cls[0].boxes:
            bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy().astype(int)
            class_id = int(box.cls[0])
            label_name = classifier.names[class_id].upper()
            conf_score = float(box.conf[0])
            
            # Kembalikan koordinat kotak ke ukuran gambar asli
            abs_x1 = bx1 + strip_x1
            abs_y1 = by1 + strip_y1
            abs_x2 = bx2 + strip_x1
            abs_y2 = by2 + strip_y1
            
            list_mata_terdeteksi.append({
                'box': [abs_x1, abs_y1, abs_x2, abs_y2],
                'label': label_name,
                'conf': conf_score
            })
            detections.append(
                Detection(label_name, conf_score, (abs_x1, abs_y1, abs_x2, abs_y2))
            )

    # Satu tempat yang memutuskan status. Kalau model tidak mengeluarkan
    # kotak, hasilnya INCONCLUSIVE tanpa angka pengganti.
    outcome = decide(detections, eye_regions_found=eye_regions_found)

    if outcome.status == STRABISMUS:
        warna = (0, 0, 255)
    elif outcome.status == NORMAL:
        warna = (0, 200, 0)
    else:
        warna = (0, 165, 255)

    # --- VISUALISASI ---
    teks = format_result(outcome)
    for d in list_mata_terdeteksi:
        x1, y1, x2, y2 = d['box']
        cv2.rectangle(img_display, (x1, y1), (x2, y2), warna, 3)
        cv2.putText(img_display, teks, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, warna, 2)

    if not list_mata_terdeteksi:
        cv2.putText(img_display, f"{teks} - {outcome.reason}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, warna, 2)

    if outcome.status == INCONCLUSIVE:
        messagebox.showwarning("Tidak dapat disimpulkan", f"{outcome.reason}.\n\n"
                               "Sistem tidak memberi label normal hanya karena tidak "
                               "menemukan apa pun.")

    cv2.imshow("Hasil Skrining Two-Stage", img_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    while True:
        process_screening()
        lagi = input("\nCoba lagi? (y/n): ").strip().lower()
        if lagi != 'y':
            break