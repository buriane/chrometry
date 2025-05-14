import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Tuple

def muat_citra(lokasi_citra: str) -> np.ndarray:
    """
    Memuat citra dalam format grayscale dari lokasi yang ditentukan.
    
    Args:
        lokasi_citra (str): Path ke file citra.
        
    Returns:
        np.ndarray: citra dalam format grayscale.
        
    Raises:
        FileNotFoundError: Jika file citra tidak ditemukan.
        ValueError: Jika citra gagal dimuat.
    """
    if not os.path.exists(lokasi_citra):
        raise FileNotFoundError(f"File citra tidak ditemukan di: {lokasi_citra}")
    
    citra = cv.imread(lokasi_citra, cv.IMREAD_GRAYSCALE)
    if citra is None:
        raise ValueError(f"Gagal memuat citra dari: {lokasi_citra}")
    
    return citra

def ubah_ukuran_citra(citra: np.ndarray, faktor_skala: float) -> np.ndarray:
    """
    Mengubah ukuran citra berdasarkan faktor skala yang diberikan.
    
    Args:
        citra (np.ndarray): citra yang akan diubah ukurannya.
        faktor_skala (float): Faktor untuk mengubah dimensi citra.
        
    Returns:
        np.ndarray: citra yang sudah diubah ukurannya.
    """
    tinggi, lebar = citra.shape
    tinggi_baru = int(tinggi * faktor_skala)
    lebar_baru = int(lebar * faktor_skala)
    return cv.resize(citra, (lebar_baru, tinggi_baru))

def proses_kontur(citra: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Memproses citra untuk mendeteksi dan mengekstrak kontur.
    
    Args:
        citra (np.ndarray): citra dalam format grayscale.
        
    Returns:
        Tuple[np.ndarray, List[np.ndarray]]: citra hasil threshold dan daftar kontur.
    """
    # Terapkan threshold untuk memisahkan objek dari latar belakang
    _, citra_threshold = cv.threshold(citra, 65, 255, cv.THRESH_BINARY)
    
    # Buat kernel untuk operasi dilasi
    kernel = np.ones((7, 7), np.uint8)
    citra_threshold = cv.dilate(citra_threshold, kernel)
    
    # Temukan kontur pada citra
    kontur, _ = cv.findContours(citra_threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    # Ambil kontur pertama (jika ada)
    return citra_threshold, [kontur[0]] if kontur else []

def visualisasi_hasil(
    citra_asli: np.ndarray,
    citra_threshold: np.ndarray,
    citra_base: np.ndarray,
    citra_proses_rgb: np.ndarray,
    kontur: List[np.ndarray],
    faktor_skala: float
) -> None:
    """
    Menampilkan hasil pemrosesan citra dalam layout 2x3 subplot.
    
    Args:
        citra_asli (np.ndarray): Citra asli dalam format grayscale.
        citra_threshold (np.ndarray): citra hasil threshold.
        citra_base (np.ndarray): citra grayscale dasar tanpa kontur berwarna.
        citra_proses_rgb (np.ndarray): citra RGB dengan kontur berwarna.
        kontur (List[np.ndarray]): Daftar kontur yang ditemukan.
        faktor_skala (float): Faktor skala yang digunakan untuk mengubah ukuran.
    """
    # Ubah ukuran citra asli agar sesuai dengan citra proses untuk perbandingan
    citra_asli_diubah = ubah_ukuran_citra(citra_asli, faktor_skala)
    
    # Buat figure dengan ukuran yang cukup besar
    plt.figure(figsize=(15, 10))
    
    # Tampilkan citra asli (diubah ukurannya agar sebanding)
    plt.subplot(231)
    plt.imshow(citra_asli_diubah, cmap='gray')
    plt.title('Citra Asli')
    plt.axis('off')
    
    # Tampilkan citra hasil threshold
    plt.subplot(232)
    plt.imshow(citra_threshold, cmap='gray')
    plt.title('Threshold')
    plt.axis('off')
    
    # Tampilkan citra dengan kontur dalam mode RGB
    plt.subplot(233)
    plt.imshow(citra_proses_rgb)
    plt.title('Kontur (Garis Hijau)')
    plt.axis('off')
    
    if not kontur:
        plt.show()
        return
    
    # Hitung dan tampilkan titik pusat serta aproksimasi poligon pada citra base
    momen = cv.moments(kontur[0])
    if momen['m00'] != 0:
        pusat_x = int(momen['m10'] / momen['m00'])
        pusat_y = int(momen['m01'] / momen['m00'])
        plt.subplot(234)
        plt.imshow(citra_base, cmap='gray')
        plt.plot(pusat_x, pusat_y, 'r*', label='Titik Pusat')
        
        # Hitung aproksimasi poligon
        keliling = cv.arcLength(kontur[0], True)
        epsilon = 0.01 * keliling
        poligon_aproksimasi = cv.approxPolyDP(kontur[0], epsilon, True)
        titik_poligon = np.array(poligon_aproksimasi)
        titik_poligon = np.concatenate((titik_poligon, titik_poligon[:1]), axis=0)
        plt.plot(titik_poligon[:, 0, 0], titik_poligon[:, 0, 1], 'b-', label='Poligon Aproksimasi')
        plt.title('Titik Pusat & Poligon')
        plt.legend()
        plt.axis('off')
    
    # Tampilkan convex hull pada citra base
    convex_hull = cv.convexHull(kontur[0])
    titik_hull = np.concatenate((convex_hull[:, 0, :], convex_hull[:1, 0, :]), axis=0)
    plt.subplot(235)
    plt.imshow(citra_base, cmap='gray')
    plt.plot(titik_hull[:, 0], titik_hull[:, 1], 'r-', label='Convex Hull')
    plt.title('Convex Hull')
    plt.legend()
    plt.axis('off')
    
    # Tampilkan kotak pembatas (bounding rectangle) pada citra base
    x, y, lebar, tinggi = cv.boundingRect(kontur[0])
    citra_dengan_kotak = citra_base.copy()
    cv.rectangle(citra_dengan_kotak, (x, y), (x + lebar, y + tinggi), (255), 2)
    plt.subplot(236)
    plt.imshow(citra_dengan_kotak, cmap='gray')
    plt.title('Kotak Pembatas')
    plt.axis('off')
    
    # Atur tata letak agar rapi
    plt.tight_layout()
    plt.show()

def analisis_kontur(kontur: np.ndarray) -> None:
    """
    Menganalisis dan mencetak properti kontur seperti luas dan keliling.
    
    Args:
        kontur (np.ndarray): Kontur yang akan dianalisis.
    """
    luas = cv.contourArea(kontur)
    keliling = cv.arcLength(kontur, True)
    print(f"Luas Kontur: {luas:.2f} piksel")
    print(f"Keliling Kontur: {keliling:.2f} piksel")

def main():
    """
    Fungsi utama untuk melakukan deteksi kontur dan visualisasi.
    """
    try:
        # Tentukan konstanta
        FOLDER_citra = 'images'
        NAMA_citra = 'tesla.png'
        FAKTOR_SKALA = 4.0
        
        # Buat path ke citra
        direktori_utama = os.getcwd()
        lokasi_citra = os.path.join(direktori_utama, FOLDER_citra, NAMA_citra)
        
        # Muat dan proses citra
        citra_asli = muat_citra(lokasi_citra)
        citra_diubah_ukuran = ubah_ukuran_citra(citra_asli, FAKTOR_SKALA)
        citra_threshold, kontur = proses_kontur(citra_diubah_ukuran)
        
        if not kontur:
            print("Tidak ada kontur yang ditemukan di citra.")
            return
        
        # Siapkan citra base (grayscale tanpa kontur berwarna)
        citra_base = citra_diubah_ukuran.copy()
        
        # Siapkan citra RGB untuk kontur berwarna
        citra_proses_rgb = cv.cvtColor(citra_diubah_ukuran, cv.COLOR_GRAY2RGB)
        cv.drawContours(citra_proses_rgb, kontur, -1, (0, 255, 0), 10)  # Hijau, ketebalan 10
        
        # Analisis dan visualisasi
        analisis_kontur(kontur[0])
        print(f"Jumlah titik dalam kontur: {len(kontur[0])}")  # Debug kontur
        visualisasi_hasil(citra_asli, citra_threshold, citra_base, citra_proses_rgb, kontur, FAKTOR_SKALA)
        
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error tak terduga: {e}")

if __name__ == "__main__":
    main()