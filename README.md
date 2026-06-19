Pipeline Restorasi & Super Resolution Citra
Aplikasi berbasis web ini dibangun menggunakan Streamlit untuk memulihkan (restorasi) foto rusak atau bergores dan meningkatkan resolusinya (Super Resolution) menggunakan kecerdasan buatan (AI).

Tim Pengembang
Aplikasi ini dikembangkan menggunakan pipeline 4 tahap oleh:

Elbert Janitra - Tahap 1: Advanced Masking (Deteksi goresan/kerusakan)

Nicholas Chandra - Tahap 2: Hybrid Inpainting (Penambalan kerusakan)

Angga Pramudya - Tahap 3: LAB Denoising (Pembersihan noise warna)

Vincent Leonardi - Tahap 4: AI Super Resolution (Peningkatan resolusi dengan FSRCNN)

Prasyarat (Prerequisites)
Pastikan Anda sudah menginstal Python (disarankan versi 3.8 ke atas) di komputer Anda. Anda bisa mengeceknya dengan membuka Terminal atau Command Prompt dan mengetikkan:

Bash
python --version
Langkah-Langkah Instalasi Library
Sangat disarankan untuk menggunakan Virtual Environment agar instalasi library tidak bentrok dengan proyek Python Anda yang lain.

1. Buat dan Aktifkan Virtual Environment (Opsional namun Disarankan)
Buka terminal/Command Prompt di dalam folder proyek Anda, lalu jalankan perintah berikut:

Untuk Windows:

Bash
python -m venv venv
venv\Scripts\activate
Untuk macOS / Linux:

Bash
python3 -m venv venv
source venv/bin/activate
2. Instalasi Dependensi / Library
Jalankan perintah pip berikut untuk menginstal semua library yang dibutuhkan oleh aplikasi:

Bash
pip install streamlit numpy opencv-contrib-python
Catatan Penting: Proyek ini menggunakan modul cv2.dnn_superres. Modul ini hanya tersedia di dalam paket opencv-contrib-python, bukan di opencv-python standar. Pastikan Anda menginstal versi contrib sesuai perintah di atas. Modul bawaan seperti os, urllib, dan ssl sudah terintegrasi dalam Python secara default, sehingga tidak perlu diinstal lagi.

Cara Menjalankan Aplikasi
Pastikan Anda berada di direktori yang sama dengan file app.py.

Jalankan perintah Streamlit berikut di terminal Anda:

Bash
streamlit run app.py
Aplikasi akan otomatis terbuka di browser default Anda (biasanya di http://localhost:8501).

Struktur & Alur Direktori
Saat aplikasi dijalankan untuk pertama kalinya, sistem akan secara otomatis membuat beberapa folder dan mengunduh model AI:

/models: Direktori ini akan dibuat otomatis. Aplikasi akan mengunduh file model AI FSRCNN_x2.pb dari GitHub secara langsung ke dalam folder ini jika belum tersedia.

/result: Setelah Anda mengunggah gambar dan menekan tombol "Eksekusi Pipeline Restorasi", hasil dari setiap tahapan (Mask, Inpainted, Denoised, Final) akan disimpan secara otomatis di dalam folder lokal ini sesuai dengan nama file aslinya.

Panduan Penggunaan
Unggah foto rusak berformat JPG, JPEG, atau PNG.

Gunakan Control Panel (Tuning) di sidebar sebelah kiri untuk menyesuaikan parameter:

Ambang Batas Goresan: Geser untuk menentukan sensitivitas deteksi kerusakan.

Lebar Deteksi Goresan: Menyesuaikan seberapa tebal area deteksi pada goresan.

Radius Penambalan: Area piksel di sekitar kerusakan yang digunakan untuk menambal.

Kekuatan Denoise: Menghaluskan noise sisa setelah penambalan.

Klik Eksekusi Pipeline Restorasi" dan tunggu hingga ke-4 proses selesai dijalankan.
