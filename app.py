import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from pathlib import Path

from screening import NORMAL, STRABISMUS, Detection, decide, format_result

st.set_page_config(page_title="Deteksi Mata Juling", layout="centered")

ROI_X_MIN_PCT = 0.2
ROI_X_MAX_PCT = 0.8
ROI_Y_MIN_PCT = 0.05
ROI_Y_MAX_PCT = 0.9

@st.cache_resource
def load_models():
    base_dir = Path(__file__).resolve().parent
    detector = YOLO(str(base_dir / "models" / "model_mata.pt"))
    classifier = YOLO(str(base_dir / "models" / "model_juling.pt"))
    return detector, classifier

@st.cache_resource
def load_cascades():
    face_casc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_casc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    return face_casc, eye_casc

def draw_dashed_rect(img, pt1, pt2, color, thickness=2, dash_length=15):
    x1, y1 = pt1
    x2, y2 = pt2
    for start in range(x1, x2, dash_length * 2):
        end = min(start + dash_length, x2)
        cv2.line(img, (start, y1), (end, y1), color, thickness)
        cv2.line(img, (start, y2), (end, y2), color, thickness)
    for start in range(y1, y2, dash_length * 2):
        end = min(start + dash_length, y2)
        cv2.line(img, (x1, start), (x1, end), color, thickness)
        cv2.line(img, (x2, start), (x2, end), color, thickness)

def create_guide_image(width=480, height=360):
    guide = np.ones((height, width, 3), dtype=np.uint8) * 230
    roi_x1 = int(ROI_X_MIN_PCT * width)
    roi_y1 = int(ROI_Y_MIN_PCT * height)
    roi_x2 = int(ROI_X_MAX_PCT * width)
    roi_y2 = int(ROI_Y_MAX_PCT * height)
    center = (width // 2, (roi_y1 + roi_y2) // 2)
    axes_w = int((roi_x2 - roi_x1) * 0.4)
    axes_h = int((roi_y2 - roi_y1) * 0.4)
    cv2.ellipse(guide, center, (axes_w, axes_h), 0, 0, 360, (180, 180, 180), 2)
    draw_dashed_rect(guide, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 200, 255), 2)
    text = "Posisikan Wajah di Area Ini"
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 0.6, 2)[0]
    text_x = (width - text_size[0]) // 2
    cv2.putText(guide, text, (text_x, roi_y2 + 30), font, 0.6, (80, 80, 80), 2)
    return guide

st.title("Aplikasi Deteksi Mata Juling")
st.markdown("""
Aplikasi ini dapat mendeteksi kondisi **Mata Juling (Strabismus)** menggunakan model AI (YOLO).
### Cara Penggunaan:
1. Pastikan wajah Anda berada di **area kotak kuning** dan menghadap **lurus** ke kamera.
2. Klik tombol **Ambil Foto** (atau **Take Photo**).
3. Sistem akan memvalidasi posisi wajah Anda terhadap area yang ditandai.
4. Jika posisi tepat, hasil klasifikasi (Juling / Normal) akan ditampilkan.
""")

detector, classifier = load_models()
face_cascade, eye_cascade = load_cascades()

source_option = st.radio("Pilih sumber gambar:", ("Kamera", "Upload Gambar"), horizontal=True)

input_image = None
if source_option == "Kamera":
    guide_img = create_guide_image()
    st.image(guide_img, caption="Panduan posisi wajah - posisikan wajah Anda di area kotak kuning", use_container_width=True)
    input_image = st.camera_input("Ambil Foto Wajah Anda")
elif source_option == "Upload Gambar":
    input_image = st.file_uploader("Upload Foto Wajah", type=['jpg', 'jpeg', 'png'])
    if input_image is not None:
        st.image(input_image, caption="Gambar yang Diupload", width=350)

if input_image is not None:
    pil_img = Image.open(input_image)
    frame = np.array(pil_img)
    
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    img_h, img_w, _ = frame_bgr.shape
    
    roi_x1 = int(ROI_X_MIN_PCT * img_w)
    roi_y1 = int(ROI_Y_MIN_PCT * img_h)
    roi_x2 = int(ROI_X_MAX_PCT * img_w)
    roi_y2 = int(ROI_Y_MAX_PCT * img_h)
    
    roi_overlay = frame_bgr.copy()
    roi_filled = roi_overlay.copy()
    cv2.rectangle(roi_filled, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 255), -1)
    roi_overlay = cv2.addWeighted(roi_filled, 0.08, roi_overlay, 0.92, 0)
    draw_dashed_rect(roi_overlay, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 255), 2)
    
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    
    face_bbox = None
    is_valid_face = False
    face_fits_roi = False
    
    if len(faces) == 0:
        st.warning("⚠️ Wajah tidak terdeteksi! Pastikan wajah Anda terlihat jelas dan berada di area kotak kuning.")
        face_bbox = (0, 0, img_w, img_h)
    elif len(faces) > 1:
        st.warning("⚠️ Terdeteksi lebih dari satu wajah! Pastikan hanya satu wajah di area kamera.")
        faces_sorted = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
        face_bbox = faces_sorted[0]
        is_valid_face = True
    else:
        face_bbox = faces[0]
        is_valid_face = True
    
    if is_valid_face:
        x, y, w, h = face_bbox
        face_center_x = x + w / 2
        face_center_y = y + h / 2
        
        face_in_roi = (x >= roi_x1) and (y >= roi_y1) and (x + w <= roi_x2) and (y + h <= roi_y2)
        center_in_roi = (roi_x1 <= face_center_x <= roi_x2) and (roi_y1 <= face_center_y <= roi_y2)
        face_fits_roi = face_in_roi or center_in_roi
        
        cv2.rectangle(roi_overlay, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.putText(roi_overlay, "Wajah", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        warnings = []
        if not face_in_roi and not center_in_roi:
            if face_center_x < roi_x1:
                warnings.append("geser wajah ke **kanan**")
            elif face_center_x > roi_x2:
                warnings.append("geser wajah ke **kiri**")
            if face_center_y < roi_y1:
                warnings.append("geser wajah ke **bawah**")
            elif face_center_y > roi_y2:
                warnings.append("geser wajah ke **atas**")
        elif not face_in_roi:
            roi_w = roi_x2 - roi_x1
            roi_h = roi_y2 - roi_y1
            if w > roi_w or h > roi_h:
                warnings.append("wajah terlalu dekat - **menjauhlah** sedikit")
            else:
                warnings.append("wajah tidak sepenuhnya di dalam area - **sesuaikan posisi** Anda")
        
        if w < 0.1 * img_w:
            warnings.append("wajah terlalu jauh - **mendekatlah** ke kamera")
        
        if warnings:
            msg = "⚠️ Wajah tidak pas di area yang ditandai: " + ", ".join(warnings) + "."
            st.warning(msg)
        else:
            st.success("✅ Posisi wajah sudah tepat di area yang ditandai!")
    
    st.image(cv2.cvtColor(roi_overlay, cv2.COLOR_BGR2RGB), caption="Area ROI (kuning) & Posisi Wajah Terdeteksi", use_container_width=True)
    
    if not face_fits_roi and is_valid_face:
        st.warning("⚠️ Hasil deteksi mungkin **kurang akurat** karena posisi wajah tidak optimal.")
    
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

        eye_regions_found = bool(regions)

        if not regions:
            # Tidak ada mata yang terdeteksi sama sekali. Stage 2 tetap
            # dijalankan pada area wajah supaya pengguna melihat hasilnya,
            # tetapi keputusan akhirnya INCONCLUSIVE, bukan NORMAL.
            if is_valid_face:
                fx, fy, fw, fh = face_bbox
                regions.append([fx, fy, fx+fw, fy+fh])
            else:
                regions.append([0, 0, img_w, img_h])

        list_mata_terdeteksi = []
        detections = []
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

        # Satu tempat yang memutuskan status. Tidak ada angka pengganti:
        # kalau model tidak mengeluarkan kotak, hasilnya INCONCLUSIVE.
        outcome = decide(detections, eye_regions_found=eye_regions_found)

        # --- VISUALISASI ---
        draw_dashed_rect(draw_img, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 255), 1)
        cv2.putText(draw_img, "ROI Wajah", (roi_x1 + 5, roi_y1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        if outcome.status == STRABISMUS:
            warna = (0, 0, 255)
        elif outcome.status == NORMAL:
            warna = (0, 200, 0)
        else:
            warna = (0, 165, 255)

        teks = format_result(outcome)
        for d in list_mata_terdeteksi:
            x1, y1, x2, y2 = d['box']
            cv2.rectangle(draw_img, (x1, y1), (x2, y2), warna, 3)
            cv2.putText(draw_img, teks, (x1, y1 - 10 if y1 > 10 else y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, warna, 2)

        if not list_mata_terdeteksi:
            cv2.putText(draw_img, teks, (roi_x1, max(roi_y2 + 30, 30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, warna, 2)

        res_rgb = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)
        st.image(res_rgb, caption="Hasil Deteksi", use_container_width=True)
                
        # Display textual results
        st.markdown("### Kesimpulan Deteksi:")
        if outcome.status == STRABISMUS:
            st.error(f"🔴 Terdeteksi **{outcome.status}** dengan tingkat keyakinan {outcome.confidence:.0%}")
        elif outcome.status == NORMAL:
            st.success(f"🟢 Terdeteksi **{outcome.status}** dengan tingkat keyakinan {outcome.confidence:.0%}")
        else:
            st.warning(
                f"⚪ **{outcome.status}** — {outcome.reason}. "
                "Sistem tidak memberi label normal hanya karena tidak menemukan apa pun."
            )
