import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from pathlib import Path

st.set_page_config(page_title="Deteksi Mata Juling", layout="centered")

@st.cache_resource
def load_models():
    base_dir = Path(__file__).resolve().parent
    detector = YOLO(str(base_dir / "models" / "model_mata.pt"))
    classifier = YOLO(str(base_dir / "models" / "model_juling.pt"))
    return detector, classifier

@st.cache_resource
def load_cascades():
    # Load OpenCV face and eye detectors
    face_casc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_casc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    return face_casc, eye_casc

st.title("Aplikasi Deteksi Mata Juling")
st.markdown("""
Aplikasi ini dapat mendeteksi kondisi **Mata Juling (Strabismus)** menggunakan model AI (YOLO).
### Cara Penggunaan:
1. Pastikan wajah Anda berada di **tengah** dan menghadap **lurus** ke kamera.
2. Klik tombol **Ambil Foto** (atau **Take Photo**).
3. Sistem akan memvalidasi posisi wajah Anda. Jika kurang tepat, akan muncul peringatan.
4. Jika posisi tepat, hasil klasifikasi (Juling / Normal) akan ditampilkan.
""")

detector, classifier = load_models()
face_cascade, eye_cascade = load_cascades()

source_option = st.radio("Pilih sumber gambar:", ("Kamera", "Upload Gambar"), horizontal=True)

input_image = None
if source_option == "Kamera":
    input_image = st.camera_input("Ambil Foto Wajah Anda")
elif source_option == "Upload Gambar":
    input_image = st.file_uploader("Upload Foto Wajah", type=['jpg', 'jpeg', 'png'])
    if input_image is not None:
        st.image(input_image, caption="Gambar yang Diupload", width=350)

if input_image is not None:
    # Convert uploaded image to OpenCV format
    pil_img = Image.open(input_image)
    frame = np.array(pil_img)
    
    # st.camera_input returns RGB, OpenCV needs BGR for standard processing
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    img_h, img_w, _ = frame_bgr.shape
    
    # Detect face to check alignment
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    
    face_bbox = None
    is_valid_face = False
    
    if len(faces) == 0:
        if source_option == "Kamera":
            st.warning("⚠️ Wajah utuh tidak terdeteksi dengan jelas, namun gambar tetap diproses.")
        # Fallback menggunakan keseluruhan gambar jika wajah tak terdeteksi
        face_bbox = (0, 0, img_w, img_h)
    elif len(faces) > 1:
        st.warning("⚠️ Terdeteksi lebih dari satu wajah! Menggunakan wajah terbesar.")
        faces_sorted = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
        face_bbox = faces_sorted[0]
        is_valid_face = True
    else:
        face_bbox = faces[0]
        is_valid_face = True
        
    if is_valid_face and source_option == "Kamera":
        x, y, w, h = face_bbox
        face_center_x = x + w / 2
        face_center_y = y + h / 2
        
        safe_x_min, safe_x_max = int(0.3 * img_w), int(0.7 * img_w)
        safe_y_min, safe_y_max = int(0.2 * img_h), int(0.8 * img_h)
        
        is_centered = (safe_x_min <= face_center_x <= safe_x_max) and (safe_y_min <= face_center_y <= safe_y_max)
        is_large_enough = (w > 0.15 * img_w)
        
        if not is_centered:
            st.warning("⚠️ Peringatan: Wajah Anda tidak di tengah! Untuk hasil lebih baik, luruskan wajah Anda.")
        if not is_large_enough:
            st.warning("⚠️ Peringatan: Wajah Anda terlalu jauh!")
            
    # Tetap jalankan proses YOLO inference
    with st.spinner("Menganalisis gambar..."):
        # --- TAHAP 1: DETEKSI AREA MATA (Menggunakan model_mata.pt) ---
        results_det = detector.predict(frame_bgr, conf=0.15, verbose=False, device='cpu')
        regions = []

        for r in results_det:
            for box in r.boxes:
                bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy().astype(int)
                regions.append([bx1, by1, bx2, by2])

        # Fallback: Jika model_mata.pt gagal, gunakan OpenCV Haar Cascade
        if not regions:
            eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (ex, ey, ew, eh) in eyes:
                x1 = max(0, ex)
                y1 = max(0, ey)
                x2 = min(img_w, ex + ew)
                y2 = min(img_h, ey + eh)
                regions.append([x1, y1, x2, y2])

        if not regions:
            # Fallback: jika mata tidak terdeteksi sama sekali, gunakan area wajah atau gambar keseluruhan
            if is_valid_face:
                fx, fy, fw, fh = face_bbox
                regions.append([fx, fy, fx+fw, fy+fh])
            else:
                regions.append([0, 0, img_w, img_h])

        status_pasien = "NORMAL"
        max_confidence = 0.0
        list_mata_terdeteksi = []
        draw_img = frame_bgr.copy()

        if regions:
            # Buat satu area besar (Horizontal Strip) yang mencakup kedua mata
            min_x = min([r[0] for r in regions])
            min_y = min([r[1] for r in regions])
            max_x = max([r[2] for r in regions])
            max_y = max([r[3] for r in regions])
            
            # Tambahkan padding agar mencakup area sekitar mata
            pad_x, pad_y = 20, 20
            strip_x1 = max(0, min_x - pad_x)
            strip_y1 = max(0, min_y - pad_y)
            strip_x2 = min(img_w, max_x + pad_x)
            strip_y2 = min(img_h, max_y + pad_y)
            
            strip_img = frame_bgr[strip_y1:strip_y2, strip_x1:strip_x2]
            
            # --- TAHAP 2: KLASIFIKASI PADA STRIP MATA ---
            results_cls = classifier.predict(strip_img, conf=0.5, verbose=False, device='cpu')
            
            boxes = results_cls[0].boxes
            if len(boxes) > 0:
                for box in boxes:
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
                    
                    # LOGIKA UNILATERAL: 1 mata juling = STRABISMUS
                    if "JULING" in label_name or "STRABISMUS" in label_name:
                        status_pasien = "STRABISMUS (JULING)"
                        if conf_score > max_confidence:
                            max_confidence = conf_score
            else:
                # Fallback jika model_juling tidak mendeteksi apa-apa di dalam strip
                for (rx1, ry1, rx2, ry2) in regions:
                    list_mata_terdeteksi.append({
                        'box': [rx1, ry1, rx2, ry2],
                        'label': "NORMAL",
                        'conf': 1.0
                    })

        # Ambil confidence tertinggi jika NORMAL
        if status_pasien == "NORMAL" and list_mata_terdeteksi:
            max_confidence = max(d['conf'] for d in list_mata_terdeteksi)

        # --- VISUALISASI ---
        for d in list_mata_terdeteksi:
            x1, y1, x2, y2 = d['box']
            warna = (0, 0, 255) if status_pasien == "STRABISMUS (JULING)" else (0, 200, 0)

            cv2.rectangle(draw_img, (x1, y1), (x2, y2), warna, 3)
            teks = f"{status_pasien} ({max_confidence:.2%})"
            cv2.putText(draw_img, teks, (x1, y1 - 10 if y1 > 10 else y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, warna, 2)

        res_rgb = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)
        st.image(res_rgb, caption="Hasil Deteksi", use_container_width=True)
                
        # Display textual results
        if not list_mata_terdeteksi:
            st.info("Area mata tidak ditemukan oleh detector.")
        else:
            st.markdown("### Kesimpulan Deteksi:")
            if status_pasien == "STRABISMUS (JULING)":
                st.error(f"🔴 Terdeteksi **{status_pasien}** dengan tingkat keyakinan {max_confidence:.0%}")
            else:
                st.success(f"🟢 Terdeteksi **{status_pasien}** dengan tingkat keyakinan {max_confidence:.0%}")
