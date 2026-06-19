import streamlit as st
import cv2
import numpy as np
import os
import urllib.request
import ssl
from cv2 import dnn_superres

# ==============================================================================
# 1. KONFIGURASI HALAMAN & PATH DIREKTORI (LOKAL)
# ==============================================================================
st.set_page_config(page_title="Pipeline Restorasi Citra", layout="wide", page_icon="⚙️")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, 'result')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'FSRCNN_x2.pb')

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ==============================================================================
# 2. FUNGSI MODULAR (LOGIKA BARU YANG LEBIH SIMPEL & PRESISI)
# ==============================================================================
@st.cache_data(show_spinner=False)
def download_model_if_not_exists(model_path):
    if not os.path.exists(model_path):
        url = "https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x2.pb"
        ssl._create_default_https_context = ssl._create_unverified_context
        try:
            urllib.request.urlretrieve(url, model_path)
        except Exception as e:
            st.error(f"Gagal mengunduh model: {e}")
    return True

# TAHAP 1 (ELBERT) - DIRUBAH MENJADI ABSOLUTE THRESHOLD
def task1_advanced_masking(img_rgb, threshold_value, kernel_size):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    # CLAHE untuk meratakan pencahayaan
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_clahe = clahe.apply(gray)

    # Karena retakan di foto rata-rata berwarna putih/terang, kita murni pakai Top-Hat
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
    tophat = cv2.morphologyEx(gray_clahe, cv2.MORPH_TOPHAT, kernel)

    # RAW ABSOLUTE CONTROL: Tidak pakai rumus mean/std lagi. Murni dari slider (0-255)
    _, mask = cv2.threshold(tophat, threshold_value, 255, cv2.THRESH_BINARY)

    # Membersihkan debu kecil (noise 1-2 piksel) agar tidak ikut ditambal
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(mask)
    for cnt in contours:
        if cv2.contourArea(cnt) > 2:  
            cv2.drawContours(filtered_mask, [cnt], -1, 255, -1)

    return cv2.dilate(filtered_mask, np.ones((3,3), np.uint8), iterations=1)

# TAHAP 2 (NICHOLAS)
def task2_hybrid_inpaint(img_rgb, mask, inpaint_radius):
    if cv2.countNonZero(mask) == 0:
        return img_rgb.copy()
    
    inpaint_telea = cv2.inpaint(img_rgb, mask, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_TELEA)
    inpaint_ns = cv2.inpaint(img_rgb, mask, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_NS)
    return cv2.addWeighted(inpaint_telea, 0.6, inpaint_ns, 0.4, 0.0)

# TAHAP 3 (ANGGA)
def task3_lab_denoising(img_rgb, bilateral_d):
    lab_image = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab_image)

    l_denoised = cv2.bilateralFilter(l, d=bilateral_d, sigmaColor=75, sigmaSpace=75)
    a_denoised = cv2.GaussianBlur(a, (3, 3), 0)
    b_denoised = cv2.GaussianBlur(b, (3, 3), 0)

    merged = cv2.merge((l_denoised, a_denoised, b_denoised))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

# TAHAP 4 (VINCENT)
def task4_ai_upscaling(img_rgb, model_path):
    gray_init = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    initial_sharpness = cv2.Laplacian(gray_init, cv2.CV_64F).var()

    sr = dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel("fsrcnn", 2)
    final_res = sr.upsample(img_rgb)

    gray_final = cv2.cvtColor(final_res, cv2.COLOR_RGB2GRAY)
    final_sharpness = cv2.Laplacian(gray_final, cv2.CV_64F).var()
    
    improvement = 0
    if initial_sharpness > 0:
        improvement = ((final_sharpness - initial_sharpness) / initial_sharpness) * 100
        
    return final_res, improvement

# ==============================================================================
# 3. ANTARMUKA PENGGUNA (UI) STREAMLIT
# ==============================================================================
with st.sidebar:
    st.markdown("## 🏛️ Universitas Bunda Mulia")
    st.markdown("### 👨‍💻 Tim Pengembang")
    st.markdown("- **Elbert Janitra** (Tahap 1)\n- **Nicholas Chandra** (Tahap 2)\n- **Angga Pramudya** (Tahap 3)\n- **Vincent Leonardi** (Tahap 4)")
    
    st.markdown("---")
    st.markdown("### 🎛️ Control Panel (Tuning)")
    st.caption("Geser angka ke KANAN untuk menyaring goresan dengan lebih selektif.")
    
    # PARAMETER BARU YANG SUPER SIMPEL (0-255)
    thresh_val = st.slider("1️⃣ Ambang Batas Goresan", min_value=10, max_value=255, value=40, step=5, help="Makin KANAN angkanya, makin SEDIKIT goresan yang terdeteksi. Cari titik pas agar wajahnya hilang dari peta putih.")
    kernel_size = st.slider("1️⃣ Lebar Deteksi Goresan", min_value=3, max_value=11, value=5, step=2)
    inpaint_rad = st.slider("2️⃣ Radius Penambalan", min_value=1, max_value=10, value=3, step=1)
    bilateral_d = st.slider("3️⃣ Kekuatan Denoise", min_value=3, max_value=15, value=5, step=2)
    
    st.markdown("---")
    st.markdown("*UAS Pengolahan Citra Digital*")

st.title("🛠️ Pipeline Restorasi & Super Resolution Citra")
st.markdown("Sistem pemrosesan gambar otomatis menggunakan metode *Morphological Masking*, *Inpainting*, dan kecerdasan buatan *FSRCNN*.")

with st.spinner("Memverifikasi Model AI..."):
    download_model_if_not_exists(MODEL_PATH)

uploaded_file = st.file_uploader("Silakan unggah foto rusak (JPG/PNG)", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    filename = uploaded_file.name
    
    st.markdown("---")
    st.subheader("Data Input")
    st.image(img_rgb, caption=f"Original Image: {filename}", width=400)
    
    if st.button("🚀 Eksekusi Pipeline Restorasi", type="primary"):
        
        st.markdown("---")
        st.subheader("Tahap 1: Advanced Masking")
        with st.spinner("Mengeksekusi analisis morfologi & kontur..."):
            mask = task1_advanced_masking(img_rgb, thresh_val, kernel_size)
            col1, col2 = st.columns(2)
            col1.image(img_rgb, caption="Input (Original)")
            col2.image(mask, caption="Output Tahap 1 (Peta Kerusakan)", clamp=True)
            
            if np.mean(mask) > 30:
                st.warning("⚠️ Masker terlihat terlalu tebal. Geser slider 'Ambang Batas Goresan' ke KANAN lalu eksekusi ulang.")
            
        st.markdown("---")
        st.subheader("Tahap 2: Hybrid Inpainting")
        with st.spinner("Menambal kerusakan dengan metode Telea & Navier-Stokes..."):
            inpainted = task2_hybrid_inpaint(img_rgb, mask, inpaint_rad)
            col1, col2 = st.columns(2)
            col1.image(mask, caption="Input (Masking)", clamp=True)
            col2.image(inpainted, caption="Output Tahap 2 (Tambalan Halus)")

        st.markdown("---")
        st.subheader("Tahap 3: LAB Denoising")
        with st.spinner("Membersihkan noda warna pada ruang LAB..."):
            denoised = task3_lab_denoising(inpainted, bilateral_d)
            col1, col2 = st.columns(2)
            col1.image(inpainted, caption="Input (Inpainted)")
            col2.image(denoised, caption="Output Tahap 3 (Bebas Noise)")

        st.markdown("---")
        st.subheader("Tahap 4: AI Super Resolution")
        with st.spinner("Meningkatkan resolusi dan ketajaman gambar dengan FSRCNN..."):
            final_res, improvement = task4_ai_upscaling(denoised, MODEL_PATH)
            
            met1, met2, met3 = st.columns(3)
            met1.metric(label="Resolusi Awal", value=f"{img_rgb.shape[1]}x{img_rgb.shape[0]} px")
            met2.metric(label="Resolusi Akhir", value=f"{final_res.shape[1]}x{final_res.shape[0]} px")
            met3.metric(label="Peningkatan Ketajaman", value=f"+{improvement:.2f}%")
            
            col1, col2 = st.columns(2)
            col1.image(denoised, caption="Input (Denoised)")
            col2.image(final_res, caption="Output Tahap 4 (Resolusi Tinggi & Tajam)")
            
        # PROSES PENYIMPANAN
        nama_tanpa_ekstensi = os.path.splitext(filename)[0]
        save_folder = os.path.join(RESULT_DIR, nama_tanpa_ekstensi)
        os.makedirs(save_folder, exist_ok=True)
        
        cv2.imwrite(os.path.join(save_folder, '1_mask.png'), mask)
        cv2.imwrite(os.path.join(save_folder, '2_inpainted.png'), cv2.cvtColor(inpainted, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(save_folder, '3_denoised.png'), cv2.cvtColor(denoised, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(save_folder, '4_final.png'), cv2.cvtColor(final_res, cv2.COLOR_RGB2BGR))
        
        st.success(f"✅ Pipeline Selesai! Semua file berhasil disimpan di folder lokal: {save_folder}")