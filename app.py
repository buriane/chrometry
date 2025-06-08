import streamlit as st
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import pandas as pd
from typing import List, Tuple, Dict
from contour import (
    muat_citra, 
    ubah_ukuran_citra, 
    proses_kontur, 
    deteksi_otomatis_parameter,
    deteksi_bentuk_geometri,
    analisis_properti_geometri
)
from color_segmentation import ColorSegmentationProcessor, create_color_segmentation_ui

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    import matplotlib.pyplot as plt
    print("Warning: Plotly not available, falling back to matplotlib")

def analisis_properti_geometri(kontur: np.ndarray) -> Dict:
    """
    Menganalisis properti geometri secara detail.
    """
    try:
        # Properti dasar
        area = cv.contourArea(kontur)
        perimeter = cv.arcLength(kontur, True)
        
        # Momen
        M = cv.moments(kontur)
        cx = int(M['m10']/M['m00']) if M['m00'] != 0 else 0
        cy = int(M['m01']/M['m00']) if M['m00'] != 0 else 0
        
        # Properti tambahan
        hull = cv.convexHull(kontur)
        hull_area = cv.contourArea(hull)
        
        # Hitung properti dengan safety checks
        x, y, w, h = cv.boundingRect(kontur)
        aspect_ratio = float(w)/h if h != 0 else 1.0
        solidity = float(area)/hull_area if hull_area > 0 else 0
        extent = float(area)/(w*h) if w*h != 0 else 0
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter != 0 else 0
        
        return {
            'area': area,
            'perimeter': perimeter,
            'center_x': cx,
            'center_y': cy,
            'solidity': solidity,
            'circularity': circularity,
            'extent': extent,
            'aspect_ratio': aspect_ratio,
            'width': w,
            'height': h
        }
    except Exception as e:
        print(f"Error in analisis_properti_geometri: {str(e)}")
        return {
            'area': 0,
            'perimeter': 0,
            'center_x': 0,
            'center_y': 0,
            'solidity': 0,
            'circularity': 0,
            'extent': 0,
            'aspect_ratio': 1.0,
            'width': 0,
            'height': 0
        }

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
    """
    try:
        plt.figure(figsize=(15, 10))
        
        # Subplot 1: Citra Asli
        citra_asli_diubah = ubah_ukuran_citra(citra_asli, faktor_skala)
        plt.subplot(231)
        plt.imshow(citra_asli_diubah, cmap='gray')
        plt.title('Citra Asli')
        plt.axis('off')
        
        # Subplot 2: Threshold
        plt.subplot(232)
        plt.imshow(citra_threshold, cmap='gray')
        plt.title('Threshold')
        plt.axis('off')
        
        # Subplot 3: Kontur (Garis Hijau)
        plt.subplot(233)
        plt.imshow(citra_proses_rgb)
        plt.title('Kontur (Garis Hijau)')
        plt.axis('off')
        
        if kontur:
            # Subplot 4: Titik Kontur
            citra_kontur_detail = citra_base.copy()
            citra_kontur_detail_rgb = cv.cvtColor(citra_kontur_detail, cv.COLOR_GRAY2RGB)
            for point in kontur[0]:
                x, y = point[0]
                cv.circle(citra_kontur_detail_rgb, (x, y), 2, (0, 0, 255), -1)  # Titik merah
            plt.subplot(234)
            plt.imshow(citra_kontur_detail_rgb)
            plt.title(f'Titik Kontur (total: {len(kontur[0])})')
            plt.axis('off')
            
            # Subplot 5: Convex Hull
            momen = cv.moments(kontur[0])
            if momen['m00'] != 0:
                pusat_x = int(momen['m10'] / momen['m00'])
                pusat_y = int(momen['m01'] / momen['m00'])
                citra_convex_hull = cv.cvtColor(citra_base.copy(), cv.COLOR_GRAY2RGB)
                
                # Buat convex hull dengan warna merah dan garis tebal
                hull = cv.convexHull(kontur[0])
                cv.polylines(citra_convex_hull, [hull], True, (255, 0, 0), 4)  # Warna merah dengan ketebalan 4
                cv.circle(citra_convex_hull, (pusat_x, pusat_y), 5, (0, 255, 0), -1)  # Titik pusat hijau
                
                plt.subplot(235)
                plt.imshow(citra_convex_hull)
                plt.title(f'Convex Hull ({len(hull)} titik)')
                plt.axis('off')
            
            # Subplot 6: Kotak Pembatas
            x, y, lebar, tinggi = cv.boundingRect(kontur[0])
            citra_dengan_kotak = citra_base.copy()
            cv.rectangle(citra_dengan_kotak, (x, y), (x + lebar, y + tinggi), (255), 2)
            plt.subplot(236)
            plt.imshow(citra_dengan_kotak, cmap='gray')
            plt.title('Kotak Pembatas')
            plt.axis('off')
        
        plt.tight_layout()
        # Save the plot properly
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        plt.savefig(temp_file.name, bbox_inches='tight', dpi=300)
        plt.close()  # Make sure to close the figure
        return temp_file.name
        
    except Exception as e:
        plt.close()  # Make sure to close in case of error
        raise e

def main():
    st.set_page_config(page_title="Aplikasi Web Pengolahan Citra", layout="wide")
    
    st.title("Aplikasi Pengolahan Citra Digital")
    st.write("Unggah citra untuk analisis deteksi kontur dan segmentasi warna.")
    
    # Penjelasan umum aplikasi
    with st.expander("Tentang Aplikasi", expanded=False):
        st.markdown("""
        Aplikasi ini menyediakan dua fitur utama pengolahan citra:
        
        **1. Deteksi Kontur dan Analisis Bentuk:**
        - Mengidentifikasi bentuk dan garis tepi objek dalam citra
        - Menghitung properti seperti luas, keliling, dan fitur geometris
        - Menampilkan visualisasi kontur, poligon aproksimasi, dan bounding box
        - Analisis properti geometri detail seperti circularity, solidity, dan extent
        
        **2. Segmentasi Warna:**
        - Ekstraksi palet warna dominan dari citra
        - Segmentasi citra berdasarkan warna dominan menggunakan K-means clustering
        - Visualisasi palet warna yang diekstrak
        - Export data palet dalam format CSV
        
        Unggah satu citra untuk mendapatkan analisis lengkap dari kedua fitur tersebut.
        """)
    
    # Parameter untuk deteksi kontur (dipindahkan ke awal)
    st.subheader("Parameter Deteksi Kontur")
    faktor_skala = st.slider("Faktor Skala", min_value=1.0, max_value=10.0, value=4.0, step=0.1, help="Atur faktor skala untuk mengubah ukuran citra")
    
    # Tombol unduh untuk contoh citra
    st.subheader("Unduh Contoh Citra")
    col1, col2, col3, col4 = st.columns(4)
    images_folder = "images"
    
    contoh_citra = [
        ("botol.jpeg", "Kernel 2", col1),
        ("pocari.png", "Kernel 1.3", col2),
        ("mouse.png", "Kernel 2.4", col3),
        ("container.jpg", "Kernel 4", col4)
    ]
    
    for image_name, kernel_info, col in contoh_citra:
        image_path = os.path.join(images_folder, image_name)
        if os.path.exists(image_path):
            col.markdown(f"**{kernel_info}**")
            with open(image_path, "rb") as file:
                col.download_button(
                    label=f"Unduh {image_name}",
                    data=file,
                    file_name=image_name,
                    mime="image/png" if image_name.endswith(".png") else "image/jpeg"
                )
        else:
            col.warning(f"File {image_name} tidak ditemukan.")
    
    # Upload gambar utama
    st.subheader("Unggah Citra")
    uploaded_file = st.file_uploader(
        "Pilih citra untuk dianalisis...", 
        type=["jpg", "jpeg", "png"], 
        help="Unggah citra dalam format JPG, JPEG, atau PNG"
    )
    
    if uploaded_file is not None:
        # ===== DETEKSI KONTUR DAN ANALISIS BENTUK =====
        st.header("🔍 Deteksi Kontur dan Analisis Bentuk")
        
        try:
            # Simpan file yang diunggah ke lokasi sementara
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name
            
            # Container untuk hasil analisis
            with st.container():
                st.subheader("Hasil Analisis Citra")
                
                with st.spinner("Memproses citra untuk deteksi kontur..."):
                    # ===== DETEKSI KONTUR =====
                    citra_asli = muat_citra(temp_file_path)
                    citra_diubah_ukuran = ubah_ukuran_citra(citra_asli, faktor_skala)
                    
                    # Definisikan kombinasi parameter seperti di contour.py
                    kombinasi_parameter = [
                        (False, 65, 3, False),
                        (False, 127, 3, False),
                        (False, 200, 3, False),
                        (False, 65, 3, True),
                        (False, 127, 5, True),
                        (True, 0, 3, False),
                        (True, 0, 3, True)
                    ]
                    
                    kontur_terbaik = []
                    citra_threshold_terbaik = None
                    jumlah_titik_terbaik = 0
                    parameter_terbaik = None
                    
                    for mode, nilai, ukuran, invert in kombinasi_parameter:
                        citra_threshold, kontur = proses_kontur(
                            citra_diubah_ukuran,
                            mode_adaptif=mode,
                            nilai_threshold=nilai,
                            ukuran_kernel_dilasi=ukuran,
                            invert_threshold=invert
                        )
                        
                        if kontur and len(kontur[0]) > jumlah_titik_terbaik:
                            jumlah_titik_terbaik = len(kontur[0])
                            kontur_terbaik = kontur
                            citra_threshold_terbaik = citra_threshold
                            parameter_terbaik = (mode, nilai, ukuran, invert)
                            
                            if jumlah_titik_terbaik > 500:
                                break
                    
                    if not kontur_terbaik:
                        st.error("Tidak ada kontur yang terdeteksi dengan semua parameter yang dicoba.")
                    else:
                        kontur = kontur_terbaik
                        citra_threshold = citra_threshold_terbaik
                        mode_adaptif, nilai_threshold, ukuran_kernel, invert_threshold = parameter_terbaik
                        st.info(f"Parameter terbaik: Mode {'Adaptif' if mode_adaptif else 'Global'}, Threshold: {nilai_threshold}, Kernel: {ukuran_kernel}x{ukuran_kernel}, Inversi: {'Ya' if invert_threshold else 'Tidak'}")
                
                # Analisis Bentuk dan Kontur
                st.subheader("Analisis Bentuk dan Kontur")
                
                if kontur:
                    # Create two columns for metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    # Use the imported deteksi_bentuk_geometri (returns 3 values)
                    shape_2d, confidence, shape_3d = deteksi_bentuk_geometri(kontur[0])
                    properties = analisis_properti_geometri(kontur[0])

                    # Display metrics
                    col1.metric("Bentuk 2D Terdeteksi", shape_2d)
                    col2.metric("Bentuk Ruang", shape_3d)
                    col3.metric("Tingkat Kepercayaan", f"{confidence:.2%}")
                    col4.metric("Luas", f"{properties['area']:.0f} px²")
                    
                    # Additional geometric properties
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Visualization of contour detection
                        citra_base = citra_diubah_ukuran.copy()
                        citra_proses_rgb = cv.cvtColor(citra_diubah_ukuran, cv.COLOR_GRAY2RGB)
                        cv.drawContours(citra_proses_rgb, kontur, -1, (0, 255, 0), 3)
                        
                        with col1:
                            # First save the plot
                            plot_path = save_plot_to_file(citra_asli, citra_threshold, citra_base, citra_proses_rgb, kontur, faktor_skala)
                            # Then display it with st.image
                            try:
                                st.image(plot_path, caption="Visualisasi Deteksi Kontur", use_column_width=True)
                            finally:
                                # Make sure we clean up the temporary file
                                if os.path.exists(plot_path):
                                    os.unlink(plot_path)
                        
                    with col2:
                        # Radar chart for shape properties
                        categories = ['Circularity', 'Solidity', 'Extent']
                        values = [properties['circularity'], properties['solidity'], properties['extent']]
                        
                        if PLOTLY_AVAILABLE:
                            fig = go.Figure(data=go.Scatterpolar(
                                r=values + [values[0]],
                                theta=categories + [categories[0]],
                                fill='toself'
                            ))
                            
                            fig.update_layout(
                                polar=dict(
                                    radialaxis=dict(
                                        visible=True,
                                        range=[0, 1]
                                    )),
                                showlegend=False,
                                title="Properti Bentuk"
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            # Fallback to matplotlib
                            fig = plt.figure(figsize=(6, 6))
                            ax = fig.add_subplot(111, polar=True)
                            ax.fill(values + [values[0]], alpha=0.25)
                            ax.set_xticks(np.linspace(0, 2 * np.pi, len(categories), endpoint=False))
                            ax.set_xticklabels(categories)
                            ax.set_yticklabels([])
                            plt.title("Properti Bentuk")
                            st.pyplot(fig)
                    
                    # Additional geometric information in an expander
                    with st.expander("Detail Properti Geometri"):
                        detail_col1, detail_col2 = st.columns(2)
                        
                        with detail_col1:
                            st.write("**Properti Dasar:**")
                            st.write(f"- Jumlah Titik Kontur: {len(kontur[0])}")
                            st.write(f"- Luas: {properties['area']:.2f} px²")
                            st.write(f"- Keliling: {properties['perimeter']:.2f} px")
                            st.write(f"- Rasio Lebar/Tinggi: {properties['aspect_ratio']:.2f}")
                            st.write(f"- Dimensi: {properties['width']}x{properties['height']} px")
                            st.write(f"- Pusat Massa: ({properties['center_x']}, {properties['center_y']})")
                        
                        with detail_col2:
                            st.write("**Properti Lanjutan:**")
                            st.write(f"- Circularity: {properties['circularity']:.3f}")
                            st.write(f"- Solidity: {properties['solidity']:.3f}")
                            st.write(f"- Extent: {properties['extent']:.3f}")
                else:
                    st.error("Tidak ada kontur yang ditemukan pada citra dengan semua parameter yang dicoba.")
            
            # ===== SEGMENTASI WARNA =====
            st.markdown("---")  # Divider line
            st.header("🎨 Segmentasi Warna (Color Segmentation)")
            st.write("Ekstraksi palet warna dan segmentasi warna menggunakan K-means clustering.")
            
            # Reset file pointer untuk segmentasi warna
            uploaded_file.seek(0)
            
            # Gunakan fungsi UI dari color_segmentation.py
            create_color_segmentation_ui(uploaded_file)
            
            os.unlink(temp_file_path)
        
        except Exception as e:
            st.error(f"Error saat memproses citra: {e}")
            import traceback
            st.error(traceback.format_exc())
            if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

if __name__ == "__main__":
    main()