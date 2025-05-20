import streamlit as st
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from contour import muat_citra, ubah_ukuran_citra, proses_kontur, deteksi_otomatis_parameter
from color_segmentation import process_color_segmentation
from typing import List, Tuple
import tempfile
import pandas as pd

def save_plot_to_file(
    citra_asli: np.ndarray,
    citra_threshold: np.ndarray,
    citra_base: np.ndarray,
    citra_proses_rgb: np.ndarray,
    kontur: List[np.ndarray],
    faktor_skala: float
) -> str:
    """
    Menyimpan plot visualisasi ke file sementara dan mengembalikan jalurnya.
    
    Args:
        citra_asli (np.ndarray): Citra grayscale asli.
        citra_threshold (np.ndarray): Citra hasil threshold.
        citra_base (np.ndarray): Citra grayscale dasar tanpa kontur berwarna.
        citra_proses_rgb (np.ndarray): Citra RGB dengan kontur berwarna.
        kontur (List[np.ndarray]): Daftar kontur yang terdeteksi.
        faktor_skala (float): Faktor skala untuk mengubah ukuran citra.
        
    Returns:
        str: Jalur ke file plot yang disimpan.
    """
    plt.figure(figsize=(15, 10))
    
    # Tampilkan citra asli (diubah ukurannya untuk perbandingan)
    citra_asli_diubah = ubah_ukuran_citra(citra_asli, faktor_skala)
    plt.subplot(231)
    plt.imshow(citra_asli_diubah, cmap='gray')
    plt.title('Citra Asli')
    plt.axis('off')
    
    # Tampilkan citra hasil threshold
    plt.subplot(232)
    plt.imshow(citra_threshold, cmap='gray')
    plt.title('Threshold')
    plt.axis('off')
    
    # Tampilkan citra dengan kontur dalam RGB
    plt.subplot(233)
    plt.imshow(citra_proses_rgb)
    plt.title('Kontur (Garis Hijau)')
    plt.axis('off')
    
    if kontur:
        # Tampilkan titik kontur dengan detail
        citra_kontur_detail = citra_base.copy()
        citra_kontur_detail_rgb = cv.cvtColor(citra_kontur_detail, cv.COLOR_GRAY2RGB)
        
        # Gambar semua titik kontur sebagai titik kecil
        for point in kontur[0]:
            x, y = point[0]
            cv.circle(citra_kontur_detail_rgb, (x, y), 2, (0, 0, 255), -1)  # Titik merah
        
        plt.subplot(234)
        plt.imshow(citra_kontur_detail_rgb)
        plt.title(f'Titik Kontur (total: {len(kontur[0])})')
        plt.axis('off')
        
        # Tampilkan titik pusat dan aproksimasi poligon
        momen = cv.moments(kontur[0])
        if momen['m00'] != 0:
            pusat_x = int(momen['m10'] / momen['m00'])
            pusat_y = int(momen['m01'] / momen['m00'])
            
            # Tampilkan aproksimasi poligon pada citra base
            citra_poligon = cv.cvtColor(citra_base.copy(), cv.COLOR_GRAY2RGB)
            
            # Hitung aproksimasi poligon
            keliling = cv.arcLength(kontur[0], True)
            epsilon = 0.01 * keliling
            poligon_aproksimasi = cv.approxPolyDP(kontur[0], epsilon, True)
            
            # Gambar poligon aproksimasi
            cv.polylines(citra_poligon, [poligon_aproksimasi], True, (0, 0, 255), 2)
            
            # Gambar titik pusat
            cv.circle(citra_poligon, (pusat_x, pusat_y), 5, (255, 0, 0), -1)
            
            plt.subplot(235)
            plt.imshow(citra_poligon)
            plt.title(f'Poligon Aproksimasi ({len(poligon_aproksimasi)} titik)')
            plt.axis('off')
        
        # Tampilkan kotak pembatas dan convex hull
        citra_bounding = cv.cvtColor(citra_base.copy(), cv.COLOR_GRAY2RGB)
        
        # Convex Hull
        hull = cv.convexHull(kontur[0])
        cv.polylines(citra_bounding, [hull], True, (255, 0, 0), 2)
        
        # Bounding Rectangle
        x, y, lebar, tinggi = cv.boundingRect(kontur[0])
        cv.rectangle(citra_bounding, (x, y), (x + lebar, y + tinggi), (0, 255, 0), 2)
        
        plt.subplot(236)
        plt.imshow(citra_bounding)
        plt.title('Kotak Pembatas & Convex Hull')
        plt.axis('off')
    
    plt.tight_layout()
    
    # Simpan plot ke file sementara
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    plt.savefig(temp_file.name, bbox_inches='tight')
    plt.close()
    return temp_file.name

def main():
    st.set_page_config(page_title="Aplikasi Web Pengolahan Citra", layout="wide")
    
    st.title("Aplikasi Pengolahan Citra Terpadu")
    st.write("Unggah citra untuk melihat hasil deteksi kontur dan segmentasi warna secara bersamaan.")
    
    # Penjelasan aplikasi
    with st.expander("Tentang Aplikasi", expanded=False):
        st.markdown("""
        Aplikasi ini menggabungkan dua teknik pengolahan citra dalam satu antarmuka yang terintegrasi:
        
        **1. Deteksi Kontur:**
        - Mengidentifikasi bentuk dan garis tepi objek dalam citra
        - Menghitung properti seperti luas, keliling, dan fitur geometris
        - Menampilkan visualisasi kontur, poligon aproksimasi, dan bounding box
        
        **2. Segmentasi Warna:**
        - Menganalisis komposisi warna dalam citra menggunakan K-Means clustering
        - Mengidentifikasi warna-warna dominan dan mengklasifikasikannya
        - Menampilkan statistik warna, distribusi kategori, dan palet warna
        
        Cukup unggah satu citra, dan Anda akan mendapatkan analisis lengkap dari kedua teknik tersebut.
        """)
    
    # Tombol unduh untuk contoh citra (semua dalam satu bagian)
    st.subheader("Unduh Contoh Citra")
    col1, col2, col3, col4 = st.columns(4)
    images_folder = "images"
    
    contoh_citra = [
        ("tesla.png", col1),
        ("tesla.jpg", col2),
        ("colorful.png", col3),
        ("sunset.jpg", col4)
    ]
    
    for image_name, col in contoh_citra:
        image_path = os.path.join(images_folder, image_name)
        if os.path.exists(image_path):
            with open(image_path, "rb") as file:
                col.download_button(
                    label=f"Unduh {image_name}",
                    data=file,
                    file_name=image_name,
                    mime="image/png" if image_name.endswith(".png") else "image/jpeg"
                )
        else:
            col.warning(f"File {image_name} tidak ditemukan.")
    
    # Pengunggah citra
    uploaded_file = st.file_uploader("Pilih citra...", type=["jpg", "jpeg", "png"], help="Unggah citra dalam format JPG, JPEG, atau PNG")
    
    # Parameter untuk kedua mode
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Parameter Deteksi Kontur")
        # Slider untuk faktor skala
        faktor_skala = st.slider("Faktor Skala", min_value=1.0, max_value=10.0, value=4.0, step=0.1, help="Atur faktor skala untuk mengubah ukuran citra")
        
        # Parameter deteksi kontur
        otomatis = st.checkbox("Deteksi Parameter Otomatis", value=True, help="Mendeteksi parameter threshold dan kernel secara otomatis berdasarkan karakteristik citra")
        
        if not otomatis:
            mode_adaptif = st.checkbox("Gunakan Threshold Adaptif", value=False, help="Menggunakan threshold adaptif untuk deteksi objek")
            nilai_threshold = st.slider("Nilai Threshold", min_value=0, max_value=255, value=65, step=1, help="Nilai ambang batas untuk threshold global")
            ukuran_kernel = st.slider("Ukuran Kernel Dilasi", min_value=1, max_value=9, value=3, step=2, help="Ukuran kernel untuk operasi dilasi")
            invert_threshold = st.checkbox("Inversi Threshold", value=False, help="Membalik mode threshold (objek terang/gelap)")
        else:
            mode_adaptif = False
            nilai_threshold = 65
            ukuran_kernel = 3
            invert_threshold = False
    
    with col2:
        st.subheader("Parameter Segmentasi Warna")
        # Parameter segmentasi warna
        auto_determine = st.checkbox("Tentukan jumlah cluster otomatis", value=True, help="Menentukan jumlah optimal cluster warna secara otomatis")
        
        n_clusters = None if auto_determine else st.slider(
            "Jumlah Cluster Warna", 
            min_value=2, 
            max_value=10, 
            value=5, 
            step=1, 
            help="Tentukan jumlah cluster warna yang diinginkan"
        )
        
        min_percentage = st.slider(
            "Persentase Minimum Warna", 
            min_value=0.1, 
            max_value=10.0, 
            value=1.0, 
            step=0.1, 
            help="Warna dengan persentase di bawah nilai ini akan diabaikan"
        )
    
    if uploaded_file is not None:
        try:
            # Simpan file yang diunggah ke lokasi sementara
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name
            
            # Container untuk hasil analisis terpadu
            with st.container():
                st.header("Hasil Analisis Citra")
                
                # Buat dua spinner terpisah agar user tahu dua proses sedang berjalan
                with st.spinner("Memproses citra..."):
                    # ===== DETEKSI KONTUR =====
                    citra_asli = muat_citra(temp_file_path)
                    citra_diubah_ukuran = ubah_ukuran_citra(citra_asli, faktor_skala)
                    
                    # Gunakan parameter otomatis jika dipilih
                    if otomatis:
                        mode_adaptif, nilai_threshold, ukuran_kernel = deteksi_otomatis_parameter(citra_diubah_ukuran)
                        
                        # Tampilkan parameter yang terdeteksi
                        st.info(f"Parameter terdeteksi otomatis: Mode {'Adaptif' if mode_adaptif else 'Global'}, Threshold: {nilai_threshold}, Kernel: {ukuran_kernel}x{ukuran_kernel}")
                    
                    # Proses kontur dengan parameter yang ditentukan
                    citra_threshold, kontur = proses_kontur(
                        citra_diubah_ukuran,
                        mode_adaptif=mode_adaptif,
                        nilai_threshold=nilai_threshold,
                        ukuran_kernel_dilasi=ukuran_kernel,
                        invert_threshold=invert_threshold
                    )
                    
                    if not kontur:
                        # Coba beberapa kombinasi parameter untuk mendapatkan kontur
                        st.warning("Tidak ada kontur yang ditemukan dengan parameter saat ini. Mencoba beberapa parameter alternatif...")
                        
                        kombinasi_parameter = [
                            # (mode_adaptif, nilai_threshold, ukuran_kernel_dilasi, invert_threshold)
                            (False, 127, 3, False),  # Threshold menengah
                            (False, 200, 3, False),  # Threshold tinggi
                            (False, 65, 3, True),    # Inverse threshold default
                            (True, 0, 3, False),     # Threshold adaptif
                            (True, 0, 3, True)       # Inverse threshold adaptif
                        ]
                        
                        for mode, nilai, ukuran, invert in kombinasi_parameter:
                            citra_threshold, kontur = proses_kontur(
                                citra_diubah_ukuran,
                                mode_adaptif=mode,
                                nilai_threshold=nilai,
                                ukuran_kernel_dilasi=ukuran,
                                invert_threshold=invert
                            )
                            
                            if kontur:
                                st.success(f"Kontur berhasil ditemukan dengan parameter: Mode {'Adaptif' if mode else 'Global'}, Threshold: {nilai}, Kernel: {ukuran}x{ukuran}, Inversi: {'Ya' if invert else 'Tidak'}")
                                break
                    
                    # ===== SEGMENTASI WARNA =====
                    # Proses citra untuk segmentasi warna
                    results, temp_files = process_color_segmentation(
                        temp_file_path,
                        n_clusters=n_clusters,
                        auto_determine=auto_determine,
                        max_clusters=8,
                        min_percentage=min_percentage
                    )
                    
                    # Dapatkan info warna dari hasil
                    color_info = results['color_info']
                    color_features = results['color_features']
                
                # Tampilkan hasil analisis setelah pemrosesan selesai
                
                # Bagian 1: Citra Original dan Hasil Deteksi Kontur
                st.subheader("1. Hasil Deteksi Kontur")
                    
                if kontur:
                    # Siapkan citra untuk visualisasi
                    citra_base = citra_diubah_ukuran.copy()
                    citra_proses_rgb = cv.cvtColor(citra_diubah_ukuran, cv.COLOR_GRAY2RGB)
                    cv.drawContours(citra_proses_rgb, kontur, -1, (0, 255, 0), 3)
                    
                    # Tampilkan statistik dasar dalam bentuk metrik
                    col1, col2, col3, col4 = st.columns(4)
                    
                    luas = cv.contourArea(kontur[0])
                    keliling = cv.arcLength(kontur[0], True)
                    jumlah_titik = len(kontur[0])
                    
                    # Hitung bentuk
                    x, y, lebar, tinggi = cv.boundingRect(kontur[0])
                    rasio_aspek = float(lebar) / tinggi if tinggi > 0 else 0
                    
                    col1.metric("Luas Kontur", f"{luas:.2f} piksel")
                    col2.metric("Keliling Kontur", f"{keliling:.2f} piksel")
                    col3.metric("Jumlah Titik", f"{jumlah_titik}")
                    col4.metric("Rasio Aspek", f"{rasio_aspek:.2f}")
                    
                    # Buat dan tampilkan plot
                    plot_path = save_plot_to_file(citra_asli, citra_threshold, citra_base, citra_proses_rgb, kontur, faktor_skala)
                    st.image(plot_path, caption="Visualisasi Deteksi Kontur", use_column_width=True)
                    
                    # Bersihkan file sementara plot
                    os.unlink(plot_path)
                else:
                    st.error("Tidak ada kontur yang ditemukan pada citra dengan semua parameter yang dicoba.")
                
                # Bagian 2: Hasil Segmentasi Warna
                st.subheader("2. Hasil Segmentasi Warna")
                
                # Tampilkan visualisasi segmentasi warna
                st.image(temp_files[0], caption="Visualisasi Segmentasi Warna", use_column_width=True)
                
                # Tampilkan info warna dalam bentuk tabel
                st.write("**Warna-Warna Dominan:**")
                st.dataframe(
                    color_info[['Kategori', 'Nama Warna', 'Persentase (%)', 'RGB', 'Hex']],
                    use_container_width=True
                )
                
                # Tampilkan palet warna dan distribusi kategori
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Palet Warna Dominan:**")
                    st.image(temp_files[1], use_column_width=True)
                
                with col2:
                    st.write("**Distribusi Kategori Warna:**")
                    st.image(temp_files[2], use_column_width=True)
                
                # Tampilkan statistik fitur warna
                st.write("**Statistik Fitur Warna:**")
                
                # Bagi statistik fitur ke dalam kategori
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("Statistik Dasar")
                    basic_features = {
                        'Kecerahan Rata-rata': color_features['brightness'],
                        'Kontras': color_features['contrast'],
                        'Colorfulness': color_features['colorfulness'],
                        'Keberagaman Warna': color_features['color_diversity'] * 100,
                        'Keseimbangan Warna': color_features['color_balance']
                    }
                    
                    basic_df = pd.DataFrame({
                        'Metrik': list(basic_features.keys()),
                        'Nilai': list(basic_features.values())
                    })
                    
                    basic_df['Nilai'] = basic_df['Nilai'].round(2)
                    st.table(basic_df)
                
                with col2:
                    st.write("Statistik RGB")
                    rgb_features = {
                        'Rata-rata Merah': color_features['mean_red'],
                        'Rata-rata Hijau': color_features['mean_green'],
                        'Rata-rata Biru': color_features['mean_blue'],
                        'Std. Dev Merah': color_features['std_red'],
                        'Std. Dev Hijau': color_features['std_green'],
                        'Std. Dev Biru': color_features['std_blue']
                    }
                    
                    rgb_df = pd.DataFrame({
                        'Metrik': list(rgb_features.keys()),
                        'Nilai': list(rgb_features.values())
                    })
                    
                    rgb_df['Nilai'] = rgb_df['Nilai'].round(2)
                    st.table(rgb_df)
                
                # Bersihkan file sementara
                for file_path in temp_files:
                    os.unlink(file_path)
            
            # Bersihkan file gambar yang diunggah
            os.unlink(temp_file_path)
        
        except Exception as e:
            st.error(f"Error saat memproses citra: {e}")
            import traceback
            st.error(traceback.format_exc())
            if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

if __name__ == "__main__":
    main()