import streamlit as st
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from contour import muat_citra, ubah_ukuran_citra, proses_kontur
from typing import List, Tuple
import tempfile

def save_plot_to_file(
    citra_asli: np.ndarray,
    citra_threshold: np.ndarray,
    citra_base: np.ndarray,
    citra_proses_rgb: np.ndarray,
    kontur: List[np.ndarray],
    faktor_skala: float
) -> str:
    """
    Save the visualization plot to a temporary file and return its path.
    
    Args:
        citra_asli (np.ndarray): Original grayscale image.
        citra_threshold (np.ndarray): Thresholded image.
        citra_base (np.ndarray): Base grayscale image without colored contours.
        citra_proses_rgb (np.ndarray): RGB image with colored contours.
        kontur (List[np.ndarray]): List of detected contours.
        faktor_skala (float): Scaling factor used for resizing.
        
    Returns:
        str: Path to the saved plot image.
    """
    plt.figure(figsize=(15, 10))
    
    # Display original image (resized for comparison)
    citra_asli_diubah = ubah_ukuran_citra(citra_asli, faktor_skala)
    plt.subplot(231)
    plt.imshow(citra_asli_diubah, cmap='gray')
    plt.title('Citra Asli')
    plt.axis('off')
    
    # Display thresholded image
    plt.subplot(232)
    plt.imshow(citra_threshold, cmap='gray')
    plt.title('Threshold')
    plt.axis('off')
    
    # Display image with contours in RGB
    plt.subplot(233)
    plt.imshow(citra_proses_rgb)
    plt.title('Kontur (Garis Hijau)')
    plt.axis('off')
    
    if kontur:
        # Display center point and polygon approximation
        momen = cv.moments(kontur[0])
        if momen['m00'] != 0:
            pusat_x = int(momen['m10'] / momen['m00'])
            pusat_y = int(momen['m01'] / momen['m00'])
            plt.subplot(234)
            plt.imshow(citra_base, cmap='gray')
            plt.plot(pusat_x, pusat_y, 'r*', label='Titik Pusat')
            
            keliling = cv.arcLength(kontur[0], True)
            epsilon = 0.01 * keliling
            poligon_aproksimasi = cv.approxPolyDP(kontur[0], epsilon, True)
            titik_poligon = np.array(poligon_aproksimasi)
            titik_poligon = np.concatenate((titik_poligon, titik_poligon[:1]), axis=0)
            plt.plot(titik_poligon[:, 0, 0], titik_poligon[:, 0, 1], 'b-', label='Poligon Aproksimasi')
            plt.title('Titik Pusat & Poligon')
            plt.legend()
            plt.axis('off')
        
        # Display convex hull
        convex_hull = cv.convexHull(kontur[0])
        titik_hull = np.concatenate((convex_hull[:, 0, :], convex_hull[:1, 0, :]), axis=0)
        plt.subplot(235)
        plt.imshow(citra_base, cmap='gray')
        plt.plot(titik_hull[:, 0], titik_hull[:, 1], 'r-', label='Convex Hull')
        plt.title('Convex Hull')
        plt.legend()
        plt.axis('off')
        
        # Display bounding rectangle
        x, y, lebar, tinggi = cv.boundingRect(kontur[0])
        citra_dengan_kotak = citra_base.copy()
        cv.rectangle(citra_dengan_kotak, (x, y), (x + lebar, y + tinggi), (255), 2)
        plt.subplot(236)
        plt.imshow(citra_dengan_kotak, cmap='gray')
        plt.title('Kotak Pembatas')
        plt.axis('off')
    
    plt.tight_layout()
    
    # Save plot to temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    plt.savefig(temp_file.name, bbox_inches='tight')
    plt.close()
    return temp_file.name

def main():
    st.title("Contour Detection Web App")
    st.write("Upload an image to detect contours and visualize the results.")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    # Slider for scaling factor
    faktor_skala = st.slider("Scaling Factor", min_value=1.0, max_value=10.0, value=4.0, step=0.1)
    
    if uploaded_file is not None:
        try:
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name
            
            # Process image using contour.py functions
            citra_asli = muat_citra(temp_file_path)
            citra_diubah_ukuran = ubah_ukuran_citra(citra_asli, faktor_skala)
            citra_threshold, kontur = proses_kontur(citra_diubah_ukuran)
            
            if not kontur:
                st.error("No contours found in the image.")
                os.unlink(temp_file_path)
                return
            
            # Prepare images for visualization
            citra_base = citra_diubah_ukuran.copy()
            citra_proses_rgb = cv.cvtColor(citra_diubah_ukuran, cv.COLOR_GRAY2RGB)
            cv.drawContours(citra_proses_rgb, kontur, -1, (0, 255, 0), 10)
            
            # Display contour analysis
            st.subheader("Contour Analysis")
            luas = cv.contourArea(kontur[0])
            keliling = cv.arcLength(kontur[0], True)
            st.write(f"**Contour Area:** {luas:.2f} pixels")
            st.write(f"**Contour Perimeter:** {keliling:.2f} pixels")
            st.write(f"**Number of Points in Contour:** {len(kontur[0])}")
            
            # Generate and display plot
            plot_path = save_plot_to_file(citra_asli, citra_threshold, citra_base, citra_proses_rgb, kontur, faktor_skala)
            st.image(plot_path, caption="Contour Detection Results", use_container_width=True)
            
            # Clean up temporary files
            os.unlink(temp_file_path)
            os.unlink(plot_path)
            
        except Exception as e:
            st.error(f"Error processing image: {e}")
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

if __name__ == "__main__":
    main()